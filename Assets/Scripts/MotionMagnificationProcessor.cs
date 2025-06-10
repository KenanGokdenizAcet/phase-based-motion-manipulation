using UnityEngine;
using System.Collections.Generic;

[RequireComponent(typeof(Camera))]
public class MotionMagnificationProcessor : MonoBehaviour
{
    [SerializeField] private ComputeShader fftComputeShader;
    [SerializeField] private ComputeShader phaseDifferenceComputeShader;
    [SerializeField] private ComputeShader pyramidOperationsShader;
    [SerializeField] private ComputeShader pyramidPhaseShader;

    [SerializeField] private bool applyMotionMagnification = true;
    [SerializeField] private bool showMagnitude = false;
    [SerializeField] private bool showPhase = false;
    
    // Pyramid parameters
    [Header("Pyramid Decomposition Settings")]
    [SerializeField] private bool usePyramidDecomposition = true;
    [SerializeField] private int pyramidLevels = 5;
    [SerializeField] [Range(0.05f, 0.45f)] private float minFrequency = 0.05f;
    [SerializeField] [Range(0.1f, 0.5f)] private float maxFrequency = 0.45f;

    // YIQ adjustment parameters
    private float yMultiplier = 1.0f;
    private float iMultiplier = 1.0f;
    private float qMultiplier = 1.0f;

    // Phase difference parameters
    [SerializeField] private float phaseScale = 10.0f; // Magnification factor !!!!!!!!
    private float magnitudeThreshold = 0.01f;
    private float magnitudeScale = 1.0f;

    // Enhanced Bandpass Filter Parameters (for non-pyramid mode)
    [Header("Phase Delta Bandpass Filter Settings")]
    [SerializeField] private bool applyBandpassFilter = true;
    [SerializeField] [Range(0.0f, 1.0f)] private float lowFrequencyCutoff = 0.05f;
    [SerializeField] [Range(0.0f, 1.0f)] private float highFrequencyCutoff = 0.4f;
    [SerializeField] [Range(0.5f, 10.0f)] private float filterSteepness = 3.0f;

    [Header("Motion Detection Enhancement")]
    [SerializeField] [Range(0.5f, 3.0f)] private float motionSensitivity = 1.5f;
    [SerializeField] private bool enhanceEdges = true;
    [SerializeField] [Range(0.0f, 2.0f)] private float edgeEnhancement = 0.8f;

    // Thread Group Size - must match compute shader
    private const int GROUP_SIZE_X = 32;
    private const int GROUP_SIZE_Y = 32;

    private int originalWidth;
    private int originalHeight;
    private int width;
    private int height;

    private Dictionary<string, RenderTexture> textures = new Dictionary<string, RenderTexture>();
    private List<RenderTexture> pyramidFilters = new List<RenderTexture>();
    private List<RenderTexture> currentPyramidLevels = new List<RenderTexture>();
    private List<RenderTexture> previousPyramidLevels = new List<RenderTexture>();
    private List<RenderTexture> processedPyramidLevels = new List<RenderTexture>();

    private ComputeBuffer complexBuffer1, complexBuffer2, previousComplexBuffer1, previousComplexBuffer2;
    private ComputeBuffer bitRevIndicesBuffer, twiddleFactorsBuffer;
    
    private Material rgbToYiqMaterial, yiqToRgbMaterial, padMaterial, cropMaterial;
    private Material windowingMaterial, blurMaterial, extractYMaterial, combineChannelsMaterial;

    private int computeBitRevIndicesKernel, computeTwiddleFactorsKernel, convertTexToComplexKernel;
    private int convertTextureToComplexKernel, convertComplexToTexRGKernel, convertComplexMagToTexKernel;
    private int convertComplexMagToTexScaledKernel, convertComplexPhaseToTexKernel, centerComplexKernel;
    private int conjugateComplexKernel, divideComplexByDimensionsKernel, bitRevByRowKernel;
    private int bitRevByColKernel, butterflyByRowKernel, butterflyByColKernel, processPhaseDifferenceKernel;

    private int generatePyramidFiltersKernel, applyPyramidFilterKernel, accumulatePyramidLevelKernel;
    private int initializeAccumulatorKernel, processPyramidPhaseDifferenceKernel;

    private bool isFirstFrame = true;
    private bool isInitialized = false;

    void OnValidate()
    {
        if (isInitialized && usePyramidDecomposition)
        {
            if (pyramidFilters.Count != pyramidLevels || pyramidFilters.Count == 0)
            {
                InitializePyramidTextures();
            }
            GeneratePyramidFilters();
        }
    }

    private void Start()
    {
        InitializeMaterials();
        InitializeProcessor();
    }

    private void OnDestroy()
    {
        ReleaseResources();
    }

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (!isInitialized || !AllShadersValid())
        {
            Graphics.Blit(source, destination);
            return;
        }

        Graphics.Blit(source, textures["sourceTexture"]);

        if (isFirstFrame)
        {
            Graphics.Blit(textures["sourceTexture"], textures["previousSourceTexture"]);
            Graphics.Blit(source, destination);
            isFirstFrame = false;
            return;
        }

        if (showMagnitude || showPhase)
        {
            ProcessDebugView(destination);
            Graphics.Blit(textures["sourceTexture"], textures["previousSourceTexture"]);
            return;
        }
        
        if (applyMotionMagnification)
        {
            if (usePyramidDecomposition)
            {
                ProcessFrameWithPyramidDecomposition(destination);
            }
            else
            {
                ProcessFrameWithStandardMagnification(destination);
            }
        }
        else
        {
            Graphics.Blit(source, destination);
        }
        
        Graphics.Blit(textures["sourceTexture"], textures["previousSourceTexture"]);
    }

    private void ProcessFrameWithPyramidDecomposition(RenderTexture destination)
    {
        Graphics.Blit(textures["sourceTexture"], textures["yiqTexture"], rgbToYiqMaterial);
        PadTexture(textures["yiqTexture"], textures["paddedTexture"]);
        ExtractYChannel(textures["paddedTexture"], textures["yChannelTexture"]);

        Graphics.Blit(textures["previousSourceTexture"], textures["previousYiqTexture"], rgbToYiqMaterial);
        PadTexture(textures["previousYiqTexture"], textures["previousPaddedTexture"]);
        ExtractYChannel(textures["previousPaddedTexture"], textures["previousYChannelTexture"]);

        PerformFFT(textures["yChannelTexture"], complexBuffer1, complexBuffer2, textures["currentDFTTexture"]);
        PerformFFT(textures["previousYChannelTexture"], previousComplexBuffer1, previousComplexBuffer2, textures["previousDFTTexture"]);

        for (int i = 0; i < pyramidLevels; i++)
        {
            pyramidOperationsShader.SetTexture(applyPyramidFilterKernel, "_InputDFT", textures["currentDFTTexture"]);
            pyramidOperationsShader.SetTexture(applyPyramidFilterKernel, "_FilterTexture", pyramidFilters[i]);
            pyramidOperationsShader.SetTexture(applyPyramidFilterKernel, "_OutputDFT", currentPyramidLevels[i]);
            DispatchCompute(pyramidOperationsShader, applyPyramidFilterKernel, width, height);

            pyramidOperationsShader.SetTexture(applyPyramidFilterKernel, "_InputDFT", textures["previousDFTTexture"]);
            pyramidOperationsShader.SetTexture(applyPyramidFilterKernel, "_FilterTexture", pyramidFilters[i]);
            pyramidOperationsShader.SetTexture(applyPyramidFilterKernel, "_OutputDFT", previousPyramidLevels[i]);
            DispatchCompute(pyramidOperationsShader, applyPyramidFilterKernel, width, height);
        }

        for (int i = 0; i < pyramidLevels; i++)
        {
            pyramidPhaseShader.SetTexture(processPyramidPhaseDifferenceKernel, "_CurrentPyramidLevel", currentPyramidLevels[i]);
            pyramidPhaseShader.SetTexture(processPyramidPhaseDifferenceKernel, "_PreviousPyramidLevel", previousPyramidLevels[i]);
            pyramidPhaseShader.SetTexture(processPyramidPhaseDifferenceKernel, "_OutputPyramidLevel", processedPyramidLevels[i]);
            pyramidPhaseShader.SetFloat("_PhaseScale", phaseScale);
            pyramidPhaseShader.SetFloat("_MagnitudeThreshold", magnitudeThreshold);
            pyramidPhaseShader.SetInt("_Width", width);
            pyramidPhaseShader.SetInt("_Height", height);
            pyramidPhaseShader.SetInt("_CurrentLevel", i);
            pyramidPhaseShader.SetInt("_TotalLevels", pyramidLevels);
            pyramidPhaseShader.SetInt("_ProcessLevel", 1);
            DispatchCompute(pyramidPhaseShader, processPyramidPhaseDifferenceKernel, width, height);
        }

        pyramidOperationsShader.SetTexture(initializeAccumulatorKernel, "_AccumulatorTexture", textures["pyramidAccumulator"]);
        DispatchCompute(pyramidOperationsShader, initializeAccumulatorKernel, width, height);

        for (int i = 0; i < pyramidLevels; i++)
        {
            pyramidOperationsShader.SetTexture(accumulatePyramidLevelKernel, "_InputDFT", processedPyramidLevels[i]);
            pyramidOperationsShader.SetTexture(accumulatePyramidLevelKernel, "_AccumulatorTexture", textures["pyramidAccumulator"]);
            DispatchCompute(pyramidOperationsShader, accumulatePyramidLevelKernel, width, height);
        }

        PerformIFFT(textures["pyramidAccumulator"], textures["processedYTexture"]);
        ApplyAntiAliasing(textures["processedYTexture"], textures["processedYTexture"]);
        CombineYIQChannels(textures["processedYTexture"], textures["paddedTexture"], textures["destinationTexture"]);
        
        yiqToRgbMaterial.SetFloat("_YMultiplier", yMultiplier);
        yiqToRgbMaterial.SetFloat("_IMultiplier", iMultiplier);
        yiqToRgbMaterial.SetFloat("_QMultiplier", qMultiplier);
        
        Graphics.Blit(textures["destinationTexture"], textures["finalTexture"], yiqToRgbMaterial);
        CropTexture(textures["finalTexture"], destination);
    }
    
    private void ProcessFrameWithStandardMagnification(RenderTexture destination)
    {
        Graphics.Blit(textures["sourceTexture"], textures["yiqTexture"], rgbToYiqMaterial);
        PadTexture(textures["yiqTexture"], textures["paddedTexture"]);
        ExtractYChannel(textures["paddedTexture"], textures["yChannelTexture"]);

        Graphics.Blit(textures["previousSourceTexture"], textures["previousYiqTexture"], rgbToYiqMaterial);
        PadTexture(textures["previousYiqTexture"], textures["previousPaddedTexture"]);
        ExtractYChannel(textures["previousPaddedTexture"], textures["previousYChannelTexture"]);

        PerformFFT(textures["yChannelTexture"], complexBuffer1, complexBuffer2, textures["currentDFTTexture"]);
        PerformFFT(textures["previousYChannelTexture"], previousComplexBuffer1, previousComplexBuffer2, textures["previousDFTTexture"]);

        ProcessPhaseDifferenceWithComputeShader(textures["currentDFTTexture"], textures["previousDFTTexture"], textures["modifiedDFTTexture"]);
        PerformIFFT(textures["modifiedDFTTexture"], textures["processedYTexture"]);
        ApplyAntiAliasing(textures["processedYTexture"], textures["processedYTexture"]);
        CombineYIQChannels(textures["processedYTexture"], textures["paddedTexture"], textures["destinationTexture"]);

        yiqToRgbMaterial.SetFloat("_YMultiplier", yMultiplier);
        yiqToRgbMaterial.SetFloat("_IMultiplier", iMultiplier);
        yiqToRgbMaterial.SetFloat("_QMultiplier", qMultiplier);

        Graphics.Blit(textures["destinationTexture"], textures["finalTexture"], yiqToRgbMaterial);
        CropTexture(textures["finalTexture"], destination);
    }
    
    private void ProcessDebugView(RenderTexture destination)
    {
        Graphics.Blit(textures["sourceTexture"], textures["yiqTexture"], rgbToYiqMaterial);
        PadTexture(textures["yiqTexture"], textures["paddedTexture"]);
        ExtractYChannel(textures["paddedTexture"], textures["yChannelTexture"]);
        PerformFFT(textures["yChannelTexture"], complexBuffer1, complexBuffer2, textures["currentDFTTexture"]);

        if (showMagnitude && !showPhase)
        {
            ConvertComplexToMagnitude(complexBuffer1, textures["magnitudeTexture"]);
            CropTexture(textures["magnitudeTexture"], destination);
        }
        else if (showPhase && !showMagnitude)
        {
            ConvertComplexToPhase(complexBuffer1, textures["phaseTexture"]);
            CropTexture(textures["phaseTexture"], destination);
        }
        else if (showMagnitude && showPhase)
        {
            ConvertComplexToMagnitude(complexBuffer1, textures["magnitudeTexture"]);
            ConvertComplexToPhase(complexBuffer1, textures["phaseTexture"]);
            ShowSplitScreen(textures["magnitudeTexture"], textures["phaseTexture"], destination);
        }
    }
    
    private void InitializeMaterials()
    {
        rgbToYiqMaterial = CreateMaterial("Custom/RGBToYIQ");
        yiqToRgbMaterial = CreateMaterial("Custom/YIQToRGB");
        padMaterial = CreateMaterial("Hidden/BlitCopy"); 
        cropMaterial = CreateMaterial("Hidden/BlitCopy");
        windowingMaterial = CreateMaterial("Hidden/WindowingFunction");
        blurMaterial = CreateMaterial("Hidden/GaussianBlur");
        extractYMaterial = CreateMaterial("Hidden/ExtractYChannel");
        combineChannelsMaterial = CreateMaterial("Hidden/CombineYIQChannels");
    }

    private Material CreateMaterial(string shaderName)
    {
        Shader shader = Shader.Find(shaderName);
        if (shader == null)
        {
            Debug.LogError($"Shader not found: {shaderName}. Please ensure it exists in the project and is included in a Resources folder or 'Always Included Shaders'.", this);
            return null;
        }
        return new Material(shader);
    }

    private bool AllShadersValid()
    {
        return rgbToYiqMaterial != null && yiqToRgbMaterial != null && padMaterial != null && cropMaterial != null &&
               windowingMaterial != null && blurMaterial != null && extractYMaterial != null && combineChannelsMaterial != null &&
               fftComputeShader != null && phaseDifferenceComputeShader != null && pyramidOperationsShader != null && pyramidPhaseShader != null;
    }

    private void InitializeProcessor()
    {
        if (!AllShadersValid())
        {
            Debug.LogError("One or more shaders are missing. Disabling component.");
            this.enabled = false;
            return;
        }
        
        originalWidth = Screen.width;
        originalHeight = Screen.height;
        int maxDimension = Mathf.Max(originalWidth, originalHeight);
        int paddedSize = Mathf.NextPowerOfTwo(maxDimension);
        width = height = paddedSize;

        Debug.Log($"Original: {originalWidth}x{originalHeight}, Padded: {width}x{height}");
        
        CreateTexture("sourceTexture", originalWidth, originalHeight, RenderTextureFormat.ARGBFloat);
        CreateTexture("previousSourceTexture", originalWidth, originalHeight, RenderTextureFormat.ARGBFloat);
        CreateTexture("paddedTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("previousPaddedTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("destinationTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("finalTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("yiqTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("previousYiqTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("yChannelTexture", width, height, RenderTextureFormat.RFloat);
        CreateTexture("previousYChannelTexture", width, height, RenderTextureFormat.RFloat);
        CreateTexture("processedYTexture", width, height, RenderTextureFormat.RFloat);
        CreateTexture("currentDFTTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("previousDFTTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("modifiedDFTTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("pyramidAccumulator", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("magnitudeTexture", width, height, RenderTextureFormat.RFloat);
        CreateTexture("phaseTexture", width, height, RenderTextureFormat.RFloat);

        InitializePyramidTextures();
        
        int complexSize = sizeof(float) * 2;
        complexBuffer1 = new ComputeBuffer(width * height, complexSize);
        complexBuffer2 = new ComputeBuffer(width * height, complexSize);
        previousComplexBuffer1 = new ComputeBuffer(width * height, complexSize);
        previousComplexBuffer2 = new ComputeBuffer(width * height, complexSize);
        bitRevIndicesBuffer = new ComputeBuffer(Mathf.Max(width, height), sizeof(int));
        twiddleFactorsBuffer = new ComputeBuffer(Mathf.Max(width, height) / 2, complexSize);

        InitializeFFTKernels();
        InitializePyramidKernels();
        if (phaseDifferenceComputeShader != null) { processPhaseDifferenceKernel = phaseDifferenceComputeShader.FindKernel("ProcessPhaseDifference"); }

        PrecomputeFFTData();
        if (usePyramidDecomposition) { GeneratePyramidFilters(); }

        isInitialized = true;
    }

    private void ReleaseResources()
    {
        ReleaseBuffer(ref complexBuffer1); ReleaseBuffer(ref complexBuffer2);
        ReleaseBuffer(ref previousComplexBuffer1); ReleaseBuffer(ref previousComplexBuffer2);
        ReleaseBuffer(ref bitRevIndicesBuffer); ReleaseBuffer(ref twiddleFactorsBuffer);
        foreach (var tex in textures.Values) if (tex != null) tex.Release();
        textures.Clear();
        ReleasePyramidTextures();
        DestroyMaterial(ref rgbToYiqMaterial); DestroyMaterial(ref yiqToRgbMaterial);
        DestroyMaterial(ref padMaterial); DestroyMaterial(ref cropMaterial);
        DestroyMaterial(ref windowingMaterial); DestroyMaterial(ref blurMaterial);
        DestroyMaterial(ref extractYMaterial); DestroyMaterial(ref combineChannelsMaterial);
    }
    
    private void PadTexture(RenderTexture source, RenderTexture destination)
    {
        float normalizedWidth = (float)originalWidth / width;
        float normalizedHeight = (float)originalHeight / height;
        float offsetX = (1.0f - normalizedWidth) * 0.5f;
        float offsetY = (1.0f - normalizedHeight) * 0.5f;
        
        RenderTexture.active = destination;
        GL.Clear(true, true, Color.black);
        GL.PushMatrix();
        GL.LoadOrtho();
        
        padMaterial.SetTexture("_MainTex", source);
        padMaterial.SetPass(0);
        
        GL.Begin(GL.QUADS);
        GL.TexCoord2(0, 0); GL.Vertex3(offsetX, offsetY, 0);
        GL.TexCoord2(1, 0); GL.Vertex3(offsetX + normalizedWidth, offsetY, 0);
        GL.TexCoord2(1, 1); GL.Vertex3(offsetX + normalizedWidth, offsetY + normalizedHeight, 0);
        GL.TexCoord2(0, 1); GL.Vertex3(offsetX, offsetY + normalizedHeight, 0);
        GL.End();

        GL.PopMatrix();
        RenderTexture.active = null;
        
        ApplyWindowingFunction(destination, destination);
    }

    private void CropTexture(RenderTexture source, RenderTexture destination)
    {
        float normalizedWidth = (float)originalWidth / width;
        float normalizedHeight = (float)originalHeight / height;
        float offsetX = (1.0f - normalizedWidth) * 0.5f;
        float offsetY = (1.0f - normalizedHeight) * 0.5f;

        RenderTexture.active = destination;
        GL.Clear(true, true, Color.black);
        GL.PushMatrix();
        GL.LoadOrtho();
        
        cropMaterial.SetTexture("_MainTex", source);
        cropMaterial.SetPass(0);

        GL.Begin(GL.QUADS);
        GL.TexCoord2(offsetX, offsetY); GL.Vertex3(0, 0, 0);
        GL.TexCoord2(offsetX + normalizedWidth, offsetY); GL.Vertex3(1, 0, 0);
        GL.TexCoord2(offsetX + normalizedWidth, offsetY + normalizedHeight); GL.Vertex3(1, 1, 0);
        GL.TexCoord2(offsetX, offsetY + normalizedHeight); GL.Vertex3(0, 1, 0);
        GL.End();
        
        GL.PopMatrix();
        RenderTexture.active = null;
    }
    
    private void ApplyWindowingFunction(RenderTexture source, RenderTexture destination)
    {
        if (windowingMaterial == null) { Graphics.Blit(source, destination); return; }
        RenderTexture temp = RenderTexture.GetTemporary(source.descriptor);
        windowingMaterial.SetFloat("_Width", width);
        windowingMaterial.SetFloat("_Height", height);
        Graphics.Blit(source, temp, windowingMaterial);
        Graphics.Blit(temp, destination);
        RenderTexture.ReleaseTemporary(temp);
    }
    
    private void ApplyAntiAliasing(RenderTexture source, RenderTexture destination)
    {
        if (blurMaterial == null) { Graphics.Blit(source, destination); return; }
        RenderTexture temp = RenderTexture.GetTemporary(source.descriptor);
        blurMaterial.SetFloat("_BlurSize", 0.5f);
        blurMaterial.SetVector("_Direction", new Vector4(1, 0, 0, 0));
        Graphics.Blit(source, temp, blurMaterial);
        blurMaterial.SetVector("_Direction", new Vector4(0, 1, 0, 0));
        Graphics.Blit(temp, destination, blurMaterial);
        RenderTexture.ReleaseTemporary(temp);
    }
    
    private void ExtractYChannel(RenderTexture yiq, RenderTexture y) { Graphics.Blit(yiq, y, extractYMaterial); }
    
    private void CombineYIQChannels(RenderTexture y, RenderTexture iq, RenderTexture output)
    {
        combineChannelsMaterial.SetTexture("_YTex", y);
        combineChannelsMaterial.SetTexture("_IQTex", iq);
        Graphics.Blit(null, output, combineChannelsMaterial);
    }
    
    private void ConvertComplexToMagnitude(ComputeBuffer buf, RenderTexture tex)
    {
        fftComputeShader.SetBuffer(convertComplexMagToTexScaledKernel, "Src", buf);
        fftComputeShader.SetTexture(convertComplexMagToTexScaledKernel, "DstTex", tex);
        DispatchCompute(fftComputeShader, convertComplexMagToTexScaledKernel, width, height);
    }
    
    private void ConvertComplexToPhase(ComputeBuffer buf, RenderTexture tex)
    {
        fftComputeShader.SetBuffer(convertComplexPhaseToTexKernel, "Src", buf);
        fftComputeShader.SetTexture(convertComplexPhaseToTexKernel, "DstTex", tex);
        DispatchCompute(fftComputeShader, convertComplexPhaseToTexKernel, width, height);
    }

    private void ShowSplitScreen(RenderTexture left, RenderTexture right, RenderTexture dest)
    {
        RenderTexture.active = dest;
        GL.Clear(true, true, Color.black);
        GL.PushMatrix();
        GL.LoadOrtho();
        
        padMaterial.SetPass(0);

        // Left
        padMaterial.SetTexture("_MainTex", left);
        GL.Begin(GL.QUADS);
        GL.TexCoord2(0, 0); GL.Vertex3(0, 0, 0);
        GL.TexCoord2(1, 0); GL.Vertex3(0.5f, 0, 0);
        GL.TexCoord2(1, 1); GL.Vertex3(0.5f, 1, 0);
        GL.TexCoord2(0, 1); GL.Vertex3(0, 1, 0);
        GL.End();

        // Right
        padMaterial.SetTexture("_MainTex", right);
        GL.Begin(GL.QUADS);
        GL.TexCoord2(0, 0); GL.Vertex3(0.5f, 0, 0);
        GL.TexCoord2(1, 0); GL.Vertex3(1, 0, 0);
        GL.TexCoord2(1, 1); GL.Vertex3(1, 1, 0);
        GL.TexCoord2(0, 1); GL.Vertex3(0.5f, 1, 0);
        GL.End();

        GL.PopMatrix();
        RenderTexture.active = null;
    }

    private void ProcessPhaseDifferenceWithComputeShader(RenderTexture currentDFT, RenderTexture previousDFT, RenderTexture outputDFT)
    {
        phaseDifferenceComputeShader.SetTexture(processPhaseDifferenceKernel, "_CurrentDFT", currentDFT);
        phaseDifferenceComputeShader.SetTexture(processPhaseDifferenceKernel, "_PreviousDFT", previousDFT);
        phaseDifferenceComputeShader.SetTexture(processPhaseDifferenceKernel, "_OutputDFT", outputDFT);
        phaseDifferenceComputeShader.SetFloat("_PhaseScale", phaseScale);
        phaseDifferenceComputeShader.SetFloat("_MagnitudeThreshold", magnitudeThreshold);
        phaseDifferenceComputeShader.SetFloat("_MagnitudeScale", magnitudeScale);
        phaseDifferenceComputeShader.SetInt("_Width", width);
        phaseDifferenceComputeShader.SetInt("_Height", height);
        phaseDifferenceComputeShader.SetBool("_ApplyBandpassFilter", applyBandpassFilter);
        phaseDifferenceComputeShader.SetFloat("_LowFreqCutoff", lowFrequencyCutoff);
        phaseDifferenceComputeShader.SetFloat("_HighFreqCutoff", highFrequencyCutoff);
        phaseDifferenceComputeShader.SetFloat("_FilterSteepness", filterSteepness);
        phaseDifferenceComputeShader.SetFloat("_MotionSensitivity", motionSensitivity);
        phaseDifferenceComputeShader.SetFloat("_EdgeEnhancement", enhanceEdges ? edgeEnhancement : 0.0f);
        DispatchCompute(phaseDifferenceComputeShader, processPhaseDifferenceKernel, width, height);
    }
    
    private void PerformFFT(RenderTexture yTexture, ComputeBuffer out1, ComputeBuffer out2, RenderTexture outTex)
    {
        fftComputeShader.SetInt("WIDTH", width); fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetTexture(convertTexToComplexKernel, "SrcTex", yTexture);
        fftComputeShader.SetBuffer(convertTexToComplexKernel, "Dst", out1);
        DispatchCompute(fftComputeShader, convertTexToComplexKernel, width, height);
        fftComputeShader.SetBuffer(centerComplexKernel, "Src", out1);
        fftComputeShader.SetBuffer(centerComplexKernel, "Dst", out2);
        DispatchCompute(fftComputeShader, centerComplexKernel, width, height);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "Src", out2);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "Dst", out1);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(fftComputeShader, bitRevByRowKernel, width, height);
        ComputeBuffer src = out1, dst = out2;
        for (int s = 2; s <= width; s *= 2)
        {
            fftComputeShader.SetInt("BUTTERFLY_STRIDE", s);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "Src", src);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "Dst", dst);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "TwiddleFactors", twiddleFactorsBuffer);
            DispatchCompute(fftComputeShader, butterflyByRowKernel, width, height);
            (src, dst) = (dst, src);
        }
        fftComputeShader.SetInt("N", height);
        fftComputeShader.SetBuffer(computeBitRevIndicesKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(fftComputeShader, computeBitRevIndicesKernel, height, 1);
        fftComputeShader.SetBuffer(computeTwiddleFactorsKernel, "TwiddleFactors", twiddleFactorsBuffer);
        DispatchCompute(fftComputeShader, computeTwiddleFactorsKernel, height / 2, 1);
        fftComputeShader.SetBuffer(bitRevByColKernel, "Src", src);
        fftComputeShader.SetBuffer(bitRevByColKernel, "Dst", dst);
        fftComputeShader.SetBuffer(bitRevByColKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(fftComputeShader, bitRevByColKernel, width, height);
        (src, dst) = (dst, src);
        for (int s = 2; s <= height; s *= 2)
        {
            fftComputeShader.SetInt("BUTTERFLY_STRIDE", s);
            fftComputeShader.SetBuffer(butterflyByColKernel, "Src", src);
            fftComputeShader.SetBuffer(butterflyByColKernel, "Dst", dst);
            fftComputeShader.SetBuffer(butterflyByColKernel, "TwiddleFactors", twiddleFactorsBuffer);
            DispatchCompute(fftComputeShader, butterflyByColKernel, width, height);
            (src, dst) = (dst, src);
        }
        fftComputeShader.SetBuffer(convertComplexToTexRGKernel, "Src", src);
        fftComputeShader.SetTexture(convertComplexToTexRGKernel, "DstTex", outTex);
        DispatchCompute(fftComputeShader, convertComplexToTexRGKernel, width, height);
    }
    
    private void ConvertTextureToBuffer(RenderTexture dftTex, ComputeBuffer outBuf)
    {
        fftComputeShader.SetInt("WIDTH", width); fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetTexture(convertTextureToComplexKernel, "SrcTex", dftTex);
        fftComputeShader.SetBuffer(convertTextureToComplexKernel, "Dst", outBuf);
        DispatchCompute(fftComputeShader, convertTextureToComplexKernel, width, height);
    }

    private void PerformIFFT(RenderTexture dftTex, RenderTexture outTex)
    {
        ConvertTextureToBuffer(dftTex, complexBuffer1);
        fftComputeShader.SetInt("WIDTH", width); fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetBuffer(conjugateComplexKernel, "Src", complexBuffer1);
        fftComputeShader.SetBuffer(conjugateComplexKernel, "Dst", complexBuffer2);
        DispatchCompute(fftComputeShader, conjugateComplexKernel, width, height);
        fftComputeShader.SetInt("N", width);
        PrecomputeFFTData();
        fftComputeShader.SetBuffer(bitRevByRowKernel, "Src", complexBuffer2);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "Dst", complexBuffer1);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(fftComputeShader, bitRevByRowKernel, width, height);
        ComputeBuffer src = complexBuffer1, dst = complexBuffer2;
        for (int s = 2; s <= width; s *= 2)
        {
            fftComputeShader.SetInt("BUTTERFLY_STRIDE", s);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "Src", src);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "Dst", dst);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "TwiddleFactors", twiddleFactorsBuffer);
            DispatchCompute(fftComputeShader, butterflyByRowKernel, width, height);
            (src, dst) = (dst, src);
        }
        fftComputeShader.SetInt("N", height);
        fftComputeShader.SetBuffer(computeBitRevIndicesKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(fftComputeShader, computeBitRevIndicesKernel, height, 1);
        fftComputeShader.SetBuffer(computeTwiddleFactorsKernel, "TwiddleFactors", twiddleFactorsBuffer);
        DispatchCompute(fftComputeShader, computeTwiddleFactorsKernel, height/2, 1);
        fftComputeShader.SetBuffer(bitRevByColKernel, "Src", src);
        fftComputeShader.SetBuffer(bitRevByColKernel, "Dst", dst);
        fftComputeShader.SetBuffer(bitRevByColKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(fftComputeShader, bitRevByColKernel, width, height);
        (src, dst) = (dst, src);
        for (int s = 2; s <= height; s *= 2)
        {
            fftComputeShader.SetInt("BUTTERFLY_STRIDE", s);
            fftComputeShader.SetBuffer(butterflyByColKernel, "Src", src);
            fftComputeShader.SetBuffer(butterflyByColKernel, "Dst", dst);
            fftComputeShader.SetBuffer(butterflyByColKernel, "TwiddleFactors", twiddleFactorsBuffer);
            DispatchCompute(fftComputeShader, butterflyByColKernel, width, height);
            (src, dst) = (dst, src);
        }
        fftComputeShader.SetBuffer(conjugateComplexKernel, "Src", src);
        fftComputeShader.SetBuffer(conjugateComplexKernel, "Dst", dst);
        DispatchCompute(fftComputeShader, conjugateComplexKernel, width, height);
        (src, dst) = (dst, src);
        fftComputeShader.SetBuffer(divideComplexByDimensionsKernel, "Src", src);
        fftComputeShader.SetBuffer(divideComplexByDimensionsKernel, "Dst", dst);
        DispatchCompute(fftComputeShader, divideComplexByDimensionsKernel, width, height);
        (src, dst) = (dst, src);
        fftComputeShader.SetBuffer(centerComplexKernel, "Src", src);
        fftComputeShader.SetBuffer(centerComplexKernel, "Dst", dst);
        DispatchCompute(fftComputeShader, centerComplexKernel, width, height);
        (src, dst) = (dst, src);
        fftComputeShader.SetBuffer(convertComplexMagToTexKernel, "Src", src);
        fftComputeShader.SetTexture(convertComplexMagToTexKernel, "DstTex", outTex);
        DispatchCompute(fftComputeShader, convertComplexMagToTexKernel, width, height);
    }
    
    private void CreateTexture(string name, int w, int h, RenderTextureFormat format)
    {
        if (textures.ContainsKey(name) && textures[name] != null) { textures[name].Release(); }
        RenderTexture tex = new RenderTexture(w, h, 0, format) { enableRandomWrite = true };
        tex.Create();
        textures[name] = tex;
    }

    private void PrecomputeFFTData()
    {
        fftComputeShader.SetInt("N", width);
        fftComputeShader.SetBuffer(computeBitRevIndicesKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(fftComputeShader, computeBitRevIndicesKernel, width, 1);
        fftComputeShader.SetBuffer(computeTwiddleFactorsKernel, "TwiddleFactors", twiddleFactorsBuffer);
        DispatchCompute(fftComputeShader, computeTwiddleFactorsKernel, width / 2, 1);
    }
    
    private void DispatchCompute(ComputeShader shader, int kernel, int w, int h)
    {
        shader.GetKernelThreadGroupSizes(kernel, out uint tx, out uint ty, out _);
        int groupsX = Mathf.CeilToInt(w / (float)tx);
        int groupsY = Mathf.CeilToInt(h / (float)ty);
        shader.Dispatch(kernel, groupsX, groupsY, 1);
    }
    
    private void InitializeFFTKernels()
    {
        computeBitRevIndicesKernel = fftComputeShader.FindKernel("ComputeBitRevIndices");
        computeTwiddleFactorsKernel = fftComputeShader.FindKernel("ComputeTwiddleFactors");
        convertTexToComplexKernel = fftComputeShader.FindKernel("ConvertTexToComplex");
        convertTextureToComplexKernel = fftComputeShader.FindKernel("ConvertTextureToComplex");
        convertComplexToTexRGKernel = fftComputeShader.FindKernel("ConvertComplexToTexRG");
        convertComplexMagToTexKernel = fftComputeShader.FindKernel("ConvertComplexMagToTex");
        convertComplexMagToTexScaledKernel = fftComputeShader.FindKernel("ConvertComplexMagToTexScaled");
        convertComplexPhaseToTexKernel = fftComputeShader.FindKernel("ConvertComplexPhaseToTex");
        centerComplexKernel = fftComputeShader.FindKernel("CenterComplex");
        conjugateComplexKernel = fftComputeShader.FindKernel("ConjugateComplex");
        divideComplexByDimensionsKernel = fftComputeShader.FindKernel("DivideComplexByDimensions");
        bitRevByRowKernel = fftComputeShader.FindKernel("BitRevByRow");
        bitRevByColKernel = fftComputeShader.FindKernel("BitRevByCol");
        butterflyByRowKernel = fftComputeShader.FindKernel("ButterflyByRow");
        butterflyByColKernel = fftComputeShader.FindKernel("ButterflyByCol");
    }

    private void InitializePyramidKernels()
    {
        generatePyramidFiltersKernel = pyramidOperationsShader.FindKernel("GeneratePyramidFilters");
        applyPyramidFilterKernel = pyramidOperationsShader.FindKernel("ApplyPyramidFilter");
        accumulatePyramidLevelKernel = pyramidOperationsShader.FindKernel("AccumulatePyramidLevel");
        initializeAccumulatorKernel = pyramidOperationsShader.FindKernel("InitializeAccumulator");
        processPyramidPhaseDifferenceKernel = pyramidPhaseShader.FindKernel("ProcessPyramidPhaseDifference");
    }

    private void InitializePyramidTextures()
    {
        ReleasePyramidTextures();
        for (int i = 0; i < pyramidLevels; i++)
        {
            pyramidFilters.Add(CreateRWTexture(width, height, RenderTextureFormat.RFloat));
            currentPyramidLevels.Add(CreateRWTexture(width, height, RenderTextureFormat.ARGBFloat));
            previousPyramidLevels.Add(CreateRWTexture(width, height, RenderTextureFormat.ARGBFloat));
            processedPyramidLevels.Add(CreateRWTexture(width, height, RenderTextureFormat.ARGBFloat));
        }
    }
    
    private RenderTexture CreateRWTexture(int w, int h, RenderTextureFormat format)
    {
        var tex = new RenderTexture(w, h, 0, format) { enableRandomWrite = true };
        tex.Create();
        return tex;
    }

    private void GeneratePyramidFilters()
    {
        if (pyramidOperationsShader == null) return;
        for (int i = 0; i < pyramidLevels; i++)
        {
            pyramidOperationsShader.SetInt("_FilterIndex", i);
            pyramidOperationsShader.SetInt("_NumFilters", pyramidLevels);
            pyramidOperationsShader.SetInt("_Width", width);
            pyramidOperationsShader.SetInt("_Height", height);
            pyramidOperationsShader.SetFloat("_MinFreq", minFrequency);
            pyramidOperationsShader.SetFloat("_MaxFreq", maxFrequency);
            pyramidOperationsShader.SetTexture(generatePyramidFiltersKernel, "_FilterTexture", pyramidFilters[i]);
            DispatchCompute(pyramidOperationsShader, generatePyramidFiltersKernel, width, height);
        }
    }
    
    private void ReleasePyramidTextures()
    {
        pyramidFilters.ForEach(t => { if(t) t.Release(); });
        currentPyramidLevels.ForEach(t => { if(t) t.Release(); });
        previousPyramidLevels.ForEach(t => { if(t) t.Release(); });
        processedPyramidLevels.ForEach(t => { if(t) t.Release(); });
        pyramidFilters.Clear();
        currentPyramidLevels.Clear();
        previousPyramidLevels.Clear();
        processedPyramidLevels.Clear();
    }

    private void ReleaseBuffer(ref ComputeBuffer buffer)
    {
        if (buffer != null) { buffer.Release(); buffer = null; }
    }

    private void DestroyMaterial(ref Material mat)
    {
        if (mat != null) { Destroy(mat); mat = null; }
    }
}