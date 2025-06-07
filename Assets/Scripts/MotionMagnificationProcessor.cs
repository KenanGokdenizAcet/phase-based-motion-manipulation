using UnityEngine;
using System.Collections.Generic;

[RequireComponent(typeof(Camera))]
public class MotionMagnificationProcessor : MonoBehaviour
{
    [SerializeField] private ComputeShader fftComputeShader;
    [SerializeField] private ComputeShader phaseDifferenceComputeShader;
    [SerializeField] private bool applyMotionMagnification = true;
    [SerializeField] private bool showMagnitude = false;
    [SerializeField] private bool showPhase = false;
    [SerializeField] private bool applyScaling = true;
    
    // YIQ adjustment parameters
    private float yMultiplier = 1.0f;
    private float iMultiplier = 1.0f;
    private float qMultiplier = 1.0f;
    
    // Phase difference parameters
    [SerializeField] private float phaseScale = 10.0f;
    private float magnitudeThreshold = 0.01f;
    private float magnitudeScale = 1.0f;

    // Enhanced Bandpass Filter Parameters
    [Header("Spatial Frequency Bandpass Filter Settings")]
    [SerializeField] private bool applyBandpassFilter = true;
    [SerializeField] [Range(0.0f, 1.0f)] private float lowFrequencyCutoff = 0.05f;
    [SerializeField] [Range(0.0f, 1.0f)] private float highFrequencyCutoff = 0.4f;
    [SerializeField] [Range(0.5f, 10.0f)] private float filterSteepness = 3.0f;
    
    [Header("Motion Detection Enhancement")]
    [SerializeField] [Range(0.5f, 3.0f)] private float motionSensitivity = 1.5f;
    [SerializeField] private bool enhanceEdges = true;
    [SerializeField] [Range(0.0f, 2.0f)] private float edgeEnhancement = 0.8f;
    
    // DC Component preservation
    [Header("DC Component Preservation")]
    [SerializeField] private bool preserveDCComponent = true;
    [SerializeField] [Range(0.0f, 0.1f)] private float dcPreservationRadius = 0.02f;
    
    // Normalization parameters
    [Header("Normalization Settings")]
    [SerializeField] private bool applyAdaptiveNormalization = true;
    [SerializeField] [Range(0.0f, 1.0f)] private float normalizationStrength = 0.8f;
    
    // Thread Group Size - must match compute shader
    private const int GROUP_SIZE_X = 32;
    private const int GROUP_SIZE_Y = 32;
    
    private int originalWidth;
    private int originalHeight;
    private int width;
    private int height;
    
    // Textures for processing
    private Dictionary<string, RenderTexture> textures = new Dictionary<string, RenderTexture>();
    
    // Compute buffers
    private ComputeBuffer complexBuffer1;
    private ComputeBuffer complexBuffer2;
    private ComputeBuffer previousComplexBuffer1;
    private ComputeBuffer previousComplexBuffer2;
    private ComputeBuffer bitRevIndicesBuffer;
    private ComputeBuffer twiddleFactorsBuffer;
    
    // Normalization buffers
    private ComputeBuffer minMaxBuffer;
    private ComputeBuffer originalMinMaxBuffer;
    
    // Materials
    private Material rgbToYiqMaterial;
    private Material yiqToRgbMaterial;
    private Material padMaterial;
    private Material cropMaterial;
    private Material windowingMaterial;
    private Material blurMaterial;
    private Material extractYMaterial;
    private Material combineChannelsMaterial;
    private Material normalizeMaterial;
    
    // Kernel IDs
    private int computeBitRevIndicesKernel;
    private int computeTwiddleFactorsKernel;
    private int convertTexToComplexKernel;
    private int convertTextureToComplexKernel;
    private int convertComplexToTexRGKernel;
    private int convertComplexMagToTexKernel;
    private int convertComplexMagToTexScaledKernel;
    private int convertComplexPhaseToTexKernel;
    private int centerComplexKernel;
    private int conjugateComplexKernel;
    private int divideComplexByDimensionsKernel;
    private int bitRevByRowKernel;
    private int bitRevByColKernel;
    private int butterflyByRowKernel;
    private int butterflyByColKernel;
    private int applyBandpassFilterKernel;
    private int processPhaseDifferenceKernel;
    private int computeMinMaxKernel;
    private int normalizeTextureKernel;
    private int applyBandpassFilterWithDCKernel;
    
    private bool isFirstFrame = true;
    private bool isInitialized = false;

    private void Start()
    {
        InitializeMaterials();
        InitializeProcessor();
    }
    
    private void InitializeMaterials()
    {
        rgbToYiqMaterial = new Material(Shader.Find("Custom/RGBToYIQ"));
        yiqToRgbMaterial = new Material(Shader.Find("Custom/YIQToRGB"));
        padMaterial = new Material(Shader.Find("Hidden/BlitCopy"));
        cropMaterial = new Material(Shader.Find("Hidden/BlitCopy"));
        windowingMaterial = CreateMaterial("Hidden/WindowingFunction");
        blurMaterial = CreateMaterial("Hidden/GaussianBlur");
        extractYMaterial = CreateMaterial("Hidden/ExtractYChannel") ?? 
                           CreateMaterial("Hidden/ExtractRedChannel");
        combineChannelsMaterial = CreateMaterial("Hidden/CombineYIQChannels") ?? 
                                 new Material(Shader.Find("Unlit/Texture"));
        normalizeMaterial = CreateMaterial("Hidden/NormalizeTexture");
    }
    
    private Material CreateMaterial(string shaderName)
    {
        Shader shader = Shader.Find(shaderName);
        return shader != null ? new Material(shader) : null;
    }

    private void OnDestroy()
    {
        ReleaseResources();
    }
    
    private void ReleaseResources()
    {
        // Release compute buffers
        ReleaseBuffer(ref complexBuffer1);
        ReleaseBuffer(ref complexBuffer2);
        ReleaseBuffer(ref previousComplexBuffer1);
        ReleaseBuffer(ref previousComplexBuffer2);
        ReleaseBuffer(ref bitRevIndicesBuffer);
        ReleaseBuffer(ref twiddleFactorsBuffer);
        ReleaseBuffer(ref minMaxBuffer);
        ReleaseBuffer(ref originalMinMaxBuffer);
        
        // Release all textures
        foreach (var texture in textures.Values)
        {
            if (texture != null)
                texture.Release();
        }
        textures.Clear();
        
        DestroyMaterial(ref rgbToYiqMaterial);
        DestroyMaterial(ref yiqToRgbMaterial);
        DestroyMaterial(ref padMaterial);
        DestroyMaterial(ref cropMaterial);
        DestroyMaterial(ref windowingMaterial);
        DestroyMaterial(ref blurMaterial);
        DestroyMaterial(ref extractYMaterial);
        DestroyMaterial(ref combineChannelsMaterial);
        DestroyMaterial(ref normalizeMaterial);
    }
    
    private void ReleaseBuffer(ref ComputeBuffer buffer)
    {
        if (buffer != null)
        {
            buffer.Release();
            buffer = null;
        }
    }
    
    private void DestroyMaterial(ref Material material)
    {
        if (material != null)
        {
            Destroy(material);
            material = null;
        }
    }

    private void InitializeProcessor()
    {
        // Get reference to the camera
        Camera cam = GetComponent<Camera>();
        if (cam == null)
        {
            Debug.LogError("Camera component is required!");
            return;
        }

        // Save original dimensions
        originalWidth = Screen.width;
        originalHeight = Screen.height;

        // Round to nearest power of 2
        int maxDimension = Mathf.Max(originalWidth, originalHeight);
        int paddedSize = Mathf.NextPowerOfTwo(maxDimension);
        
        width = paddedSize;
        height = paddedSize;

        Debug.Log($"Original size: {originalWidth}x{originalHeight}, Padded size: {width}x{height}");
        
        // Create textures
        CreateTexture("sourceTexture", originalWidth, originalHeight, RenderTextureFormat.ARGBFloat);
        CreateTexture("previousSourceTexture", originalWidth, originalHeight, RenderTextureFormat.ARGBFloat);
        CreateTexture("paddedTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("previousPaddedTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("destinationTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("finalTexture", originalWidth, originalHeight, RenderTextureFormat.ARGBFloat);
        
        // YIQ textures
        CreateTexture("yiqTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("previousYiqTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("yChannelTexture", width, height, RenderTextureFormat.RFloat);
        CreateTexture("previousYChannelTexture", width, height, RenderTextureFormat.RFloat);
        CreateTexture("processedYTexture", width, height, RenderTextureFormat.RFloat);
        CreateTexture("normalizedYTexture", width, height, RenderTextureFormat.RFloat);
        
        // FFT result textures
        CreateTexture("currentDFTTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("previousDFTTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("modifiedDFTTexture", width, height, RenderTextureFormat.ARGBFloat);
        
        // Debug textures
        CreateTexture("magnitudeTexture", width, height, RenderTextureFormat.RFloat);
        CreateTexture("phaseTexture", width, height, RenderTextureFormat.RFloat);

        // Create buffers
        int complexSize = sizeof(float) * 2;
        complexBuffer1 = new ComputeBuffer(width * height, complexSize);
        complexBuffer2 = new ComputeBuffer(width * height, complexSize);
        previousComplexBuffer1 = new ComputeBuffer(width * height, complexSize);
        previousComplexBuffer2 = new ComputeBuffer(width * height, complexSize);
        
        bitRevIndicesBuffer = new ComputeBuffer(Mathf.Max(width, height), sizeof(int));
        twiddleFactorsBuffer = new ComputeBuffer(Mathf.Max(width, height) / 2, complexSize);
        
        // Normalization buffers (min, max values)
        minMaxBuffer = new ComputeBuffer(2, sizeof(float));
        originalMinMaxBuffer = new ComputeBuffer(2, sizeof(float));

        // Get kernel IDs from FFT compute shader
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
        applyBandpassFilterKernel = fftComputeShader.FindKernel("ApplyBandpassFilter");
        
        // Try to find enhanced kernels
        try
        {
            applyBandpassFilterWithDCKernel = fftComputeShader.FindKernel("ApplyBandpassFilterWithDC");
            computeMinMaxKernel = fftComputeShader.FindKernel("ComputeMinMax");
            normalizeTextureKernel = fftComputeShader.FindKernel("NormalizeTexture");
        }
        catch
        {
            Debug.LogWarning("Enhanced kernels not found in compute shader. Using fallback methods.");
            applyBandpassFilterWithDCKernel = applyBandpassFilterKernel;
        }

        // Get kernel ID from phase difference compute shader
        if (phaseDifferenceComputeShader != null)
        {
            processPhaseDifferenceKernel = phaseDifferenceComputeShader.FindKernel("ProcessPhaseDifference");
        }
        else
        {
            Debug.LogError("Phase Difference Compute Shader is not assigned!");
        }

        // Precompute FFT data
        PrecomputeFFTData();

        isInitialized = true;
    }
    
    private void CreateTexture(string name, int width, int height, RenderTextureFormat format)
    {
        RenderTexture texture = new RenderTexture(width, height, 0, format);
        texture.enableRandomWrite = true;
        texture.Create();
        textures[name] = texture;
    }
    
    private void PrecomputeFFTData()
    {
        fftComputeShader.SetInt("N", width);
        fftComputeShader.SetBuffer(computeBitRevIndicesKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(computeBitRevIndicesKernel, width, 1);

        fftComputeShader.SetBuffer(computeTwiddleFactorsKernel, "TwiddleFactors", twiddleFactorsBuffer);
        DispatchCompute(computeTwiddleFactorsKernel, width / 2, 1);
    }
    
    private void DispatchCompute(int kernelIndex, int width, int height)
    {
        uint x, y, z;
        fftComputeShader.GetKernelThreadGroupSizes(kernelIndex, out x, out y, out z);
        
        int groupsX = Mathf.CeilToInt(width / (float)x);
        int groupsY = Mathf.CeilToInt(height / (float)y);
        
        fftComputeShader.Dispatch(kernelIndex, groupsX, groupsY, 1);
    }

    private void PadTexture(RenderTexture source, RenderTexture destination)
    {
        float normalizedWidth = (float)originalWidth / width;
        float normalizedHeight = (float)originalHeight / height;
        
        float offsetX = (1.0f - normalizedWidth) * 0.5f;
        float offsetY = (1.0f - normalizedHeight) * 0.5f;
        
        padMaterial.SetTexture("_MainTex", source);
        RenderTexture.active = destination;
        GL.Clear(true, true, Color.black);
        
        GL.PushMatrix();
        GL.LoadOrtho();
        
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
        
        cropMaterial.SetTexture("_MainTex", source);
        RenderTexture.active = destination;
        GL.Clear(true, true, Color.black);
        
        GL.PushMatrix();
        GL.LoadOrtho();
        
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
        if (windowingMaterial == null)
        {
            Graphics.Blit(source, destination);
            return;
        }
        
        if (source == destination)
        {
            RenderTexture tempRT = RenderTexture.GetTemporary(source.width, source.height, 0, source.format);
            
            windowingMaterial.SetTexture("_MainTex", source);
            windowingMaterial.SetFloat("_Width", width);
            windowingMaterial.SetFloat("_Height", height);
            
            Graphics.Blit(source, tempRT, windowingMaterial);
            Graphics.Blit(tempRT, destination);
            
            RenderTexture.ReleaseTemporary(tempRT);
        }
        else
        {
            windowingMaterial.SetTexture("_MainTex", source);
            windowingMaterial.SetFloat("_Width", width);
            windowingMaterial.SetFloat("_Height", height);
            
            Graphics.Blit(source, destination, windowingMaterial);
        }
    }
    
    private void ApplyAntiAliasing(RenderTexture source, RenderTexture destination)
    {
        if (blurMaterial == null)
        {
            Graphics.Blit(source, destination);
            return;
        }
        
        blurMaterial.SetFloat("_BlurSize", 0.5f);
        
        RenderTexture tempRT = RenderTexture.GetTemporary(source.width, source.height, 0, source.format);
        
        blurMaterial.SetVector("_Direction", new Vector4(1.0f, 0.0f, 0.0f, 0.0f));
        Graphics.Blit(source, tempRT, blurMaterial);
        
        blurMaterial.SetVector("_Direction", new Vector4(0.0f, 1.0f, 0.0f, 0.0f));
        Graphics.Blit(tempRT, destination, blurMaterial);
        
        RenderTexture.ReleaseTemporary(tempRT);
    }
    
    private void ExtractYChannel(RenderTexture yiqTexture, RenderTexture yChannelTexture)
    {
        if (extractYMaterial == null)
        {
            Debug.LogError("No suitable shader found for Y channel extraction!");
            return;
        }
        
        extractYMaterial.SetTexture("_MainTex", yiqTexture);
        Graphics.Blit(yiqTexture, yChannelTexture, extractYMaterial);
    }
    
    private void CombineYIQChannels(RenderTexture processedYTexture, RenderTexture originalYIQTexture, RenderTexture outputTexture)
    {
        if (combineChannelsMaterial == null)
        {
            Debug.LogError("No suitable shader found for YIQ channel combination!");
            return;
        }
        
        combineChannelsMaterial.SetTexture("_YTex", processedYTexture);
        combineChannelsMaterial.SetTexture("_IQTex", originalYIQTexture);
        
        Graphics.Blit(null, outputTexture, combineChannelsMaterial);
    }
    
    // Enhanced spatial frequency domain bandpass filter with DC preservation
    private void ApplyBandpassFilter(ComputeBuffer inputBuffer, ComputeBuffer outputBuffer)
    {
        if (!applyBandpassFilter)
        {
            if (inputBuffer != outputBuffer)
            {
                int size = width * height;
                int threadsPerGroup = 64;
                int numGroups = Mathf.CeilToInt(size / (float)threadsPerGroup);
                
                fftComputeShader.SetBuffer(conjugateComplexKernel, "Src", inputBuffer);
                fftComputeShader.SetBuffer(conjugateComplexKernel, "Dst", outputBuffer);
                fftComputeShader.Dispatch(conjugateComplexKernel, numGroups, 1, 1);
            }
            return;
        }
        
        // Use enhanced kernel if available
        int kernelToUse = (applyBandpassFilterWithDCKernel != applyBandpassFilterKernel && preserveDCComponent) 
            ? applyBandpassFilterWithDCKernel 
            : applyBandpassFilterKernel;
        
        // Enhanced frequency domain filtering parameters
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetFloat("_LowFreqCutoff", lowFrequencyCutoff);
        fftComputeShader.SetFloat("_HighFreqCutoff", highFrequencyCutoff);
        fftComputeShader.SetFloat("_FilterSteepness", filterSteepness);
        
        // DC preservation parameters
        fftComputeShader.SetBool("_PreserveDC", preserveDCComponent);
        fftComputeShader.SetFloat("_DCRadius", dcPreservationRadius);
        
        // Additional motion enhancement parameters
        fftComputeShader.SetFloat("_MotionSensitivity", motionSensitivity);
        fftComputeShader.SetFloat("_EdgeEnhancement", enhanceEdges ? edgeEnhancement : 0.0f);
        
        fftComputeShader.SetBuffer(kernelToUse, "Src", inputBuffer);
        fftComputeShader.SetBuffer(kernelToUse, "Dst", outputBuffer);
        DispatchCompute(kernelToUse, width, height);
    }
    
    private void ConvertComplexBufferToTexture(ComputeBuffer complexBuffer, RenderTexture outputTexture)
    {
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetBuffer(convertComplexMagToTexKernel, "Src", complexBuffer);
        fftComputeShader.SetTexture(convertComplexMagToTexKernel, "DstTex", outputTexture);
        DispatchCompute(convertComplexMagToTexKernel, width, height);
    }
    
    // Store original Y channel statistics for normalization
    private void ComputeOriginalMinMax(RenderTexture yChannelTexture)
    {
        if (computeMinMaxKernel == 0) return;
        
        fftComputeShader.SetTexture(computeMinMaxKernel, "InputTex", yChannelTexture);
        fftComputeShader.SetBuffer(computeMinMaxKernel, "MinMaxBuffer", originalMinMaxBuffer);
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        DispatchCompute(computeMinMaxKernel, 1, 1);
    }
    
    // Apply adaptive normalization to preserve original brightness range
    private void ApplyAdaptiveNormalization(RenderTexture inputTexture, RenderTexture outputTexture)
    {
        if (!applyAdaptiveNormalization || normalizeTextureKernel == 0)
        {
            Graphics.Blit(inputTexture, outputTexture);
            return;
        }
        
        // Compute current min/max
        fftComputeShader.SetTexture(computeMinMaxKernel, "InputTex", inputTexture);
        fftComputeShader.SetBuffer(computeMinMaxKernel, "MinMaxBuffer", minMaxBuffer);
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        DispatchCompute(computeMinMaxKernel, 1, 1);
        
        // Apply normalization
        fftComputeShader.SetTexture(normalizeTextureKernel, "InputTex", inputTexture);
        fftComputeShader.SetTexture(normalizeTextureKernel, "OutputTex", outputTexture);
        fftComputeShader.SetBuffer(normalizeTextureKernel, "CurrentMinMax", minMaxBuffer);
        fftComputeShader.SetBuffer(normalizeTextureKernel, "TargetMinMax", originalMinMaxBuffer);
        fftComputeShader.SetFloat("_NormalizationStrength", normalizationStrength);
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        DispatchCompute(normalizeTextureKernel, width, height);
    }

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (!isInitialized || fftComputeShader == null)
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

        // Debug modes
        if (showMagnitude || showPhase)
        {
            Graphics.Blit(textures["sourceTexture"], textures["yiqTexture"], rgbToYiqMaterial);
            PadTexture(textures["yiqTexture"], textures["paddedTexture"]);
            ExtractYChannel(textures["paddedTexture"], textures["yChannelTexture"]);
            
            PerformFFT(textures["yChannelTexture"], complexBuffer1, complexBuffer2, textures["currentDFTTexture"]);
            
            if (showMagnitude && !showPhase)
            {
                ConvertComplexToMagnitude(complexBuffer1, textures["magnitudeTexture"]);
                CropTexture(textures["magnitudeTexture"], destination);
                Graphics.Blit(textures["sourceTexture"], textures["previousSourceTexture"]);
                return;
            }
            else if (showPhase && !showMagnitude)
            {
                ConvertComplexToPhase(complexBuffer1, textures["phaseTexture"]);
                CropTexture(textures["phaseTexture"], destination);
                Graphics.Blit(textures["sourceTexture"], textures["previousSourceTexture"]);
                return;
            }
            else if (showMagnitude && showPhase)
            {
                ConvertComplexToMagnitude(complexBuffer1, textures["magnitudeTexture"]);
                ConvertComplexToPhase(complexBuffer1, textures["phaseTexture"]);
                ShowSplitScreen(textures["magnitudeTexture"], textures["phaseTexture"], destination);
                Graphics.Blit(textures["sourceTexture"], textures["previousSourceTexture"]);
                return;
            }
        }

        if (applyMotionMagnification)
        {
            ProcessFrameWithMotionMagnification(source, destination);
        }
        else
        {
            Graphics.Blit(source, destination);
            Graphics.Blit(textures["sourceTexture"], textures["previousSourceTexture"]);
        }
    }
    
    private void ConvertComplexToMagnitude(ComputeBuffer complexBuffer, RenderTexture magnitudeTexture)
    {
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetBuffer(convertComplexMagToTexScaledKernel, "Src", complexBuffer);
        fftComputeShader.SetTexture(convertComplexMagToTexScaledKernel, "DstTex", magnitudeTexture);
        DispatchCompute(convertComplexMagToTexScaledKernel, width, height);
    }
    
    private void ConvertComplexToPhase(ComputeBuffer complexBuffer, RenderTexture phaseTexture)
    {
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetBuffer(convertComplexPhaseToTexKernel, "Src", complexBuffer);
        fftComputeShader.SetTexture(convertComplexPhaseToTexKernel, "DstTex", phaseTexture);
        DispatchCompute(convertComplexPhaseToTexKernel, width, height);
    }
    
    private void ShowSplitScreen(RenderTexture magnitudeTexture, RenderTexture phaseTexture, RenderTexture destination)
    {
        RenderTexture.active = destination;
        GL.Clear(true, true, Color.black);
        
        GL.PushMatrix();
        GL.LoadOrtho();
        
        // Left half - Magnitude
        padMaterial.SetTexture("_MainTex", magnitudeTexture);
        padMaterial.SetPass(0);
        GL.Begin(GL.QUADS);
        GL.TexCoord2(0, 0); GL.Vertex3(0, 0, 0);
        GL.TexCoord2(1, 0); GL.Vertex3(0.5f, 0, 0);
        GL.TexCoord2(1, 1); GL.Vertex3(0.5f, 1, 0);
        GL.TexCoord2(0, 1); GL.Vertex3(0, 1, 0);
        GL.End();
        
        // Right half - Phase
        padMaterial.SetTexture("_MainTex", phaseTexture);
        padMaterial.SetPass(0);
        GL.Begin(GL.QUADS);
        GL.TexCoord2(0, 0); GL.Vertex3(0.5f, 0, 0);
        GL.TexCoord2(1, 0); GL.Vertex3(1, 0, 0);
        GL.TexCoord2(1, 1); GL.Vertex3(1, 1, 0);
        GL.TexCoord2(0, 1); GL.Vertex3(0.5f, 1, 0);
        GL.End();
        
        GL.PopMatrix();
        RenderTexture.active = null;
    }
    
    // Enhanced phase difference processing with magnitude preservation
    private void ProcessPhaseDifferenceWithComputeShader(RenderTexture currentDFT, RenderTexture previousDFT, RenderTexture outputDFT)
    {
        if (phaseDifferenceComputeShader == null)
        {
            Debug.LogError("Phase Difference Compute Shader is not assigned!");
            Graphics.Blit(currentDFT, outputDFT);
            return;
        }
        
        // Set parameters for the phase difference shader
        phaseDifferenceComputeShader.SetTexture(processPhaseDifferenceKernel, "_CurrentDFT", currentDFT);
        phaseDifferenceComputeShader.SetTexture(processPhaseDifferenceKernel, "_PreviousDFT", previousDFT);
        phaseDifferenceComputeShader.SetTexture(processPhaseDifferenceKernel, "_OutputDFT", outputDFT);
        
        phaseDifferenceComputeShader.SetFloat("_PhaseScale", phaseScale);
        phaseDifferenceComputeShader.SetFloat("_MagnitudeThreshold", magnitudeThreshold);
        phaseDifferenceComputeShader.SetFloat("_MagnitudeScale", magnitudeScale);
        phaseDifferenceComputeShader.SetInt("_Width", width);
        phaseDifferenceComputeShader.SetInt("_Height", height);
        
        // Enable phase-only processing (preserve magnitude)
        phaseDifferenceComputeShader.SetBool("_PreserveMagnitude", true);
        
        // Dispatch compute shader
        int groupsX = Mathf.CeilToInt(width / (float)GROUP_SIZE_X);
        int groupsY = Mathf.CeilToInt(height / (float)GROUP_SIZE_Y);
        
        phaseDifferenceComputeShader.Dispatch(processPhaseDifferenceKernel, groupsX, groupsY, 1);
    }
    
    private void ProcessFrameWithMotionMagnification(RenderTexture source, RenderTexture destination)
    {
        // Convert current frame to YIQ
        Graphics.Blit(textures["sourceTexture"], textures["yiqTexture"], rgbToYiqMaterial);
        PadTexture(textures["yiqTexture"], textures["paddedTexture"]);
        ExtractYChannel(textures["paddedTexture"], textures["yChannelTexture"]);
        
        // Store original Y channel statistics before processing
        ComputeOriginalMinMax(textures["yChannelTexture"]);
        
        // Convert previous frame to YIQ
        Graphics.Blit(textures["previousSourceTexture"], textures["previousYiqTexture"], rgbToYiqMaterial);
        PadTexture(textures["previousYiqTexture"], textures["previousPaddedTexture"]);
        ExtractYChannel(textures["previousPaddedTexture"], textures["previousYChannelTexture"]);
        
        // Process with FFT (includes spatial frequency bandpass filtering with DC preservation)
        PerformFFT(textures["yChannelTexture"], complexBuffer1, complexBuffer2, textures["currentDFTTexture"]);
        PerformFFT(textures["previousYChannelTexture"], previousComplexBuffer1, previousComplexBuffer2, textures["previousDFTTexture"]);
        
        // Use compute shader for phase difference processing (phase-only, preserves magnitude)
        ProcessPhaseDifferenceWithComputeShader(textures["currentDFTTexture"], textures["previousDFTTexture"], textures["modifiedDFTTexture"]);
        
        // Perform IFFT on the modified DFT texture
        PerformIFFT(textures["modifiedDFTTexture"], textures["processedYTexture"]);
        
        // Apply adaptive normalization to preserve original brightness range
        ApplyAdaptiveNormalization(textures["processedYTexture"], textures["normalizedYTexture"]);
        
        // Apply anti-aliasing
        ApplyAntiAliasing(textures["normalizedYTexture"], textures["normalizedYTexture"]);
        
        // Combine channels
        CombineYIQChannels(textures["normalizedYTexture"], textures["paddedTexture"], textures["destinationTexture"]);
        
        // Set YIQ adjustment parameters
        yiqToRgbMaterial.SetFloat("_YMultiplier", yMultiplier);
        yiqToRgbMaterial.SetFloat("_IMultiplier", iMultiplier);
        yiqToRgbMaterial.SetFloat("_QMultiplier", qMultiplier);
        
        // Convert back to RGB
        Graphics.Blit(textures["destinationTexture"], textures["finalTexture"], yiqToRgbMaterial);
        
        // Crop to original size
        CropTexture(textures["finalTexture"], destination);
        
        // Store current frame for next iteration
        Graphics.Blit(textures["sourceTexture"], textures["previousSourceTexture"]);
    }

    // FFT Implementation with Enhanced Bandpass Filtering and DC Preservation
    private void PerformFFT(RenderTexture yTexture, ComputeBuffer outputBuffer1, ComputeBuffer outputBuffer2, RenderTexture outputTexture)
    {
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);

        fftComputeShader.SetTexture(convertTexToComplexKernel, "SrcTex", yTexture);
        fftComputeShader.SetBuffer(convertTexToComplexKernel, "Dst", outputBuffer1);
        DispatchCompute(convertTexToComplexKernel, width, height);

        fftComputeShader.SetBuffer(centerComplexKernel, "Src", outputBuffer1);
        fftComputeShader.SetBuffer(centerComplexKernel, "Dst", outputBuffer2);
        DispatchCompute(centerComplexKernel, width, height);

        fftComputeShader.SetBuffer(bitRevByRowKernel, "Src", outputBuffer2);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "Dst", outputBuffer1);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(bitRevByRowKernel, width, height);

        ComputeBuffer src = outputBuffer1;
        ComputeBuffer dst = outputBuffer2;
        
        for (int stride = 2; stride <= width; stride *= 2)
        {
            fftComputeShader.SetInt("BUTTERFLY_STRIDE", stride);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "Src", src);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "Dst", dst);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "TwiddleFactors", twiddleFactorsBuffer);
            DispatchCompute(butterflyByRowKernel, width, height);
            
            ComputeBuffer temp = src;
            src = dst;
            dst = temp;
        }

        fftComputeShader.SetInt("N", height);
        fftComputeShader.SetBuffer(computeBitRevIndicesKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(computeBitRevIndicesKernel, height, 1);

        fftComputeShader.SetBuffer(computeTwiddleFactorsKernel, "TwiddleFactors", twiddleFactorsBuffer);
        DispatchCompute(computeTwiddleFactorsKernel, height / 2, 1);

        fftComputeShader.SetBuffer(bitRevByColKernel, "Src", src);
        fftComputeShader.SetBuffer(bitRevByColKernel, "Dst", dst);
        fftComputeShader.SetBuffer(bitRevByColKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(bitRevByColKernel, width, height);

        ComputeBuffer temp2 = src;
        src = dst;
        dst = temp2;

        for (int stride = 2; stride <= height; stride *= 2)
        {
            fftComputeShader.SetInt("BUTTERFLY_STRIDE", stride);
            fftComputeShader.SetBuffer(butterflyByColKernel, "Src", src);
            fftComputeShader.SetBuffer(butterflyByColKernel, "Dst", dst);
            fftComputeShader.SetBuffer(butterflyByColKernel, "TwiddleFactors", twiddleFactorsBuffer);
            DispatchCompute(butterflyByColKernel, width, height);
            
            ComputeBuffer temp = src;
            src = dst;
            dst = temp;
        }
        
        // Apply enhanced spatial frequency domain bandpass filter with DC preservation
        if (applyBandpassFilter)
        {
            ApplyBandpassFilter(src, dst);
            ComputeBuffer tempAfterFilter = src;
            src = dst;
            dst = tempAfterFilter;
        }

        fftComputeShader.SetBuffer(convertComplexToTexRGKernel, "Src", src);
        fftComputeShader.SetTexture(convertComplexToTexRGKernel, "DstTex", outputTexture);
        DispatchCompute(convertComplexToTexRGKernel, width, height);
    }
    
    private void ConvertTextureToBuffer(RenderTexture dftTexture, ComputeBuffer outputBuffer)
    {
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetTexture(convertTextureToComplexKernel, "SrcTex", dftTexture);
        fftComputeShader.SetBuffer(convertTextureToComplexKernel, "Dst", outputBuffer);
        DispatchCompute(convertTextureToComplexKernel, width, height);
    }
    
    private void PerformIFFT(RenderTexture dftTexture, RenderTexture outputTexture)
    {
        ConvertTextureToBuffer(dftTexture, complexBuffer1);
        
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetBuffer(conjugateComplexKernel, "Src", complexBuffer1);
        fftComputeShader.SetBuffer(conjugateComplexKernel, "Dst", complexBuffer2);
        DispatchCompute(conjugateComplexKernel, width, height);
        
        fftComputeShader.SetInt("N", width);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "Src", complexBuffer2);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "Dst", complexBuffer1);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(bitRevByRowKernel, width, height);

        ComputeBuffer src = complexBuffer1;
        ComputeBuffer dst = complexBuffer2;
        
        for (int stride = 2; stride <= width; stride *= 2)
        {
            fftComputeShader.SetInt("BUTTERFLY_STRIDE", stride);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "Src", src);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "Dst", dst);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "TwiddleFactors", twiddleFactorsBuffer);
            DispatchCompute(butterflyByRowKernel, width, height);
            
            ComputeBuffer temp = src;
            src = dst;
            dst = temp;
        }

        fftComputeShader.SetInt("N", height);
        fftComputeShader.SetBuffer(bitRevByColKernel, "Src", src);
        fftComputeShader.SetBuffer(bitRevByColKernel, "Dst", dst);
        fftComputeShader.SetBuffer(bitRevByColKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(bitRevByColKernel, width, height);

        ComputeBuffer temp2 = src;
        src = dst;
        dst = temp2;

        for (int stride = 2; stride <= height; stride *= 2)
        {
            fftComputeShader.SetInt("BUTTERFLY_STRIDE", stride);
            fftComputeShader.SetBuffer(butterflyByColKernel, "Src", src);
            fftComputeShader.SetBuffer(butterflyByColKernel, "Dst", dst);
            fftComputeShader.SetBuffer(butterflyByColKernel, "TwiddleFactors", twiddleFactorsBuffer);
            DispatchCompute(butterflyByColKernel, width, height);
            
            ComputeBuffer temp = src;
            src = dst;
            dst = temp;
        }

        fftComputeShader.SetBuffer(conjugateComplexKernel, "Src", src);
        fftComputeShader.SetBuffer(conjugateComplexKernel, "Dst", dst);
        DispatchCompute(conjugateComplexKernel, width, height);

        ComputeBuffer temp3 = src;
        src = dst;
        dst = temp3;

        fftComputeShader.SetBuffer(divideComplexByDimensionsKernel, "Src", src);
        fftComputeShader.SetBuffer(divideComplexByDimensionsKernel, "Dst", dst);
        DispatchCompute(divideComplexByDimensionsKernel, width, height);

        ComputeBuffer temp4 = src;
        src = dst;
        dst = temp4;

        fftComputeShader.SetBuffer(centerComplexKernel, "Src", src);
        fftComputeShader.SetBuffer(centerComplexKernel, "Dst", dst);
        DispatchCompute(centerComplexKernel, width, height);

        fftComputeShader.SetBuffer(convertComplexMagToTexKernel, "Src", dst);
        fftComputeShader.SetTexture(convertComplexMagToTexKernel, "DstTex", outputTexture);
        DispatchCompute(convertComplexMagToTexKernel, width, height);
    }
}