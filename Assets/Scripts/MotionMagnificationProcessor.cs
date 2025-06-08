using UnityEngine;
using System.Collections.Generic;

[RequireComponent(typeof(Camera))]
public class MotionMagnificationProcessor : MonoBehaviour
{
    [Header("Compute Shaders")]
    [SerializeField] private ComputeShader fftComputeShader;
    [SerializeField] private ComputeShader phaseDifferenceComputeShader;
    [SerializeField] private ComputeShader pyramidFiltersComputeShader;
    [SerializeField] private ComputeShader pyramidDecomposeComputeShader;
    [SerializeField] private ComputeShader pyramidReconstructComputeShader;
    
    [Header("Processing Mode")]
    [SerializeField] private bool useSteerablePyramid = false;
    [SerializeField] private bool applyMotionMagnification = true;
    [SerializeField] private bool showMagnitude = false;
    [SerializeField] private bool showPhase = false;
    [SerializeField] private bool showPyramidLevel = false;
    [SerializeField] private int pyramidLevelToShow = 0;
    [SerializeField] private bool applyScaling = true;
    
    [Header("Steerable Pyramid Parameters")]
    [SerializeField] [Range(1, 6)] private int pyramidLevels = 4;
    [SerializeField] [Range(1, 4)] private int numOrientations = 4;
    [SerializeField] [Range(0.1f, 2.0f)] private float pyramidSigma = 0.67f;
    [SerializeField] private bool includeHighPassInReconstruction = false;
    
    // YIQ adjustment parameters
    private float yMultiplier = 1.0f;
    private float iMultiplier = 1.0f;
    private float qMultiplier = 1.0f;
    
    // Phase difference parameters
    [SerializeField] private float phaseScale = 10.0f;
    private float magnitudeThreshold = 0.01f;
    private float magnitudeScale = 1.0f;

    // Enhanced Bandpass Filter Parameters - now applied to phase delta
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
    
    // Textures for processing
    private Dictionary<string, RenderTexture> textures = new Dictionary<string, RenderTexture>();
    
    // Pyramid-specific textures
    private List<RenderTexture> currentPyramidLevels = new List<RenderTexture>();
    private List<RenderTexture> previousPyramidLevels = new List<RenderTexture>();
    private List<RenderTexture> processedPyramidLevels = new List<RenderTexture>();
    private List<RenderTexture> pyramidFilters = new List<RenderTexture>();
    
    // Compute buffers
    private ComputeBuffer complexBuffer1;
    private ComputeBuffer complexBuffer2;
    private ComputeBuffer previousComplexBuffer1;
    private ComputeBuffer previousComplexBuffer2;
    private ComputeBuffer bitRevIndicesBuffer;
    private ComputeBuffer twiddleFactorsBuffer;
    
    // Materials
    private Material rgbToYiqMaterial;
    private Material yiqToRgbMaterial;
    private Material padMaterial;
    private Material cropMaterial;
    private Material windowingMaterial;
    private Material blurMaterial;
    private Material extractYMaterial;
    private Material combineChannelsMaterial;
    
    // Kernel IDs for existing shaders
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
    private int processPhaseDifferenceKernel;
    
    // Kernel IDs for pyramid shaders
    private int generateLowPassFilterKernel;
    private int generateHighPassFilterKernel;
    private int generateBandPassFilterKernel;
    private int decomposeLevelKernel;
    private int extractResidualLowPassKernel;
    private int extractResidualHighPassKernel;
    private int initializeReconstructionKernel;
    private int processPhaseDifferencePerLevelKernel;
    private int accumulatePyramidLevelKernel;
    private int addResidualsKernel;
    
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
        
        // Release all textures
        foreach (var texture in textures.Values)
        {
            if (texture != null)
                texture.Release();
        }
        textures.Clear();
        
        // Release pyramid textures
        ReleasePyramidTextures();
        
        DestroyMaterial(ref rgbToYiqMaterial);
        DestroyMaterial(ref yiqToRgbMaterial);
        DestroyMaterial(ref padMaterial);
        DestroyMaterial(ref cropMaterial);
        DestroyMaterial(ref windowingMaterial);
        DestroyMaterial(ref blurMaterial);
        DestroyMaterial(ref extractYMaterial);
        DestroyMaterial(ref combineChannelsMaterial);
    }
    
    private void ReleasePyramidTextures()
    {
        foreach (var tex in currentPyramidLevels)
            if (tex != null) tex.Release();
        foreach (var tex in previousPyramidLevels)
            if (tex != null) tex.Release();
        foreach (var tex in processedPyramidLevels)
            if (tex != null) tex.Release();
        foreach (var tex in pyramidFilters)
            if (tex != null) tex.Release();
            
        currentPyramidLevels.Clear();
        previousPyramidLevels.Clear();
        processedPyramidLevels.Clear();
        pyramidFilters.Clear();
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
        
        // FFT result textures
        CreateTexture("currentDFTTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("previousDFTTexture", width, height, RenderTextureFormat.ARGBFloat);
        CreateTexture("modifiedDFTTexture", width, height, RenderTextureFormat.ARGBFloat);
        
        // Pyramid-specific textures
        CreateTexture("pyramidReconstructionDFT", width, height, RenderTextureFormat.RGFloat);
        CreateTexture("lowPassResidualDFT", width, height, RenderTextureFormat.RGFloat);
        CreateTexture("highPassResidualDFT", width, height, RenderTextureFormat.RGFloat);
        
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

        // Initialize kernel IDs
        InitializeKernelIDs();
        
        // Initialize pyramid structures if enabled
        if (useSteerablePyramid)
        {
            InitializePyramidStructures();
        }

        // Precompute FFT data
        PrecomputeFFTData();

        isInitialized = true;
    }
    
    private void InitializeKernelIDs()
    {
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

        // Get kernel ID from phase difference compute shader
        if (phaseDifferenceComputeShader != null)
        {
            processPhaseDifferenceKernel = phaseDifferenceComputeShader.FindKernel("ProcessPhaseDifference");
        }
        
        // Get kernel IDs from pyramid shaders
        if (pyramidFiltersComputeShader != null)
        {
            generateLowPassFilterKernel = pyramidFiltersComputeShader.FindKernel("GenerateLowPassFilter");
            generateHighPassFilterKernel = pyramidFiltersComputeShader.FindKernel("GenerateHighPassFilter");
            generateBandPassFilterKernel = pyramidFiltersComputeShader.FindKernel("GenerateBandPassFilter");
        }
        
        if (pyramidDecomposeComputeShader != null)
        {
            decomposeLevelKernel = pyramidDecomposeComputeShader.FindKernel("DecomposeLevel");
            extractResidualLowPassKernel = pyramidDecomposeComputeShader.FindKernel("ExtractResidualLowPass");
            extractResidualHighPassKernel = pyramidDecomposeComputeShader.FindKernel("ExtractResidualHighPass");
        }
        
        if (pyramidReconstructComputeShader != null)
        {
            initializeReconstructionKernel = pyramidReconstructComputeShader.FindKernel("InitializeReconstruction");
            processPhaseDifferencePerLevelKernel = pyramidReconstructComputeShader.FindKernel("ProcessPhaseDifferencePerLevel");
            accumulatePyramidLevelKernel = pyramidReconstructComputeShader.FindKernel("AccumulatePyramidLevel");
            addResidualsKernel = pyramidReconstructComputeShader.FindKernel("AddResiduals");
        }
    }
    
    private void InitializePyramidStructures()
    {
        // Clear existing pyramid textures
        ReleasePyramidTextures();
        
        // Create pyramid level textures
        for (int i = 0; i < pyramidLevels; i++)
        {
            // Store multiple orientations in RGBA channels
            RenderTexture currentLevel = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat);
            currentLevel.enableRandomWrite = true;
            currentLevel.Create();
            currentPyramidLevels.Add(currentLevel);
            
            RenderTexture previousLevel = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat);
            previousLevel.enableRandomWrite = true;
            previousLevel.Create();
            previousPyramidLevels.Add(previousLevel);
            
            RenderTexture processedLevel = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat);
            processedLevel.enableRandomWrite = true;
            processedLevel.Create();
            processedPyramidLevels.Add(processedLevel);
        }
        
        // Create filter textures
        CreateTexture("pyramidLowPassFilter", width, height, RenderTextureFormat.RFloat);
        CreateTexture("pyramidHighPassFilter", width, height, RenderTextureFormat.RFloat);
        
        for (int i = 0; i < pyramidLevels; i++)
        {
            RenderTexture bandPassFilter = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat);
            bandPassFilter.enableRandomWrite = true;
            bandPassFilter.Create();
            pyramidFilters.Add(bandPassFilter);
        }
        
        // Generate pyramid filters
        GeneratePyramidFilters();
    }
    
    private void GeneratePyramidFilters()
    {
        if (pyramidFiltersComputeShader == null) return;
        
        // Set common parameters
        pyramidFiltersComputeShader.SetInt("_Width", width);
        pyramidFiltersComputeShader.SetInt("_Height", height);
        pyramidFiltersComputeShader.SetInt("_NumOrientations", numOrientations);
        pyramidFiltersComputeShader.SetFloat("_Sigma", pyramidSigma);
        
        // Generate high-pass filter
        pyramidFiltersComputeShader.SetTexture(generateHighPassFilterKernel, "_HighPassFilter", textures["pyramidHighPassFilter"]);
        DispatchCompute(pyramidFiltersComputeShader, generateHighPassFilterKernel, width, height);
        
        // Generate filters for each pyramid level
        for (int i = 0; i < pyramidLevels; i++)
        {
            pyramidFiltersComputeShader.SetInt("_ScaleIndex", i);
            
            // Generate low-pass filter for this scale
            pyramidFiltersComputeShader.SetTexture(generateLowPassFilterKernel, "_LowPassFilter", textures["pyramidLowPassFilter"]);
            DispatchCompute(pyramidFiltersComputeShader, generateLowPassFilterKernel, width, height);
            
            // Generate band-pass filters (oriented)
            pyramidFiltersComputeShader.SetTexture(generateBandPassFilterKernel, "_BandPassFilter", pyramidFilters[i]);
            DispatchCompute(pyramidFiltersComputeShader, generateBandPassFilterKernel, width, height);
        }
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
        DispatchCompute(fftComputeShader, computeBitRevIndicesKernel, width, 1);

        fftComputeShader.SetBuffer(computeTwiddleFactorsKernel, "TwiddleFactors", twiddleFactorsBuffer);
        DispatchCompute(fftComputeShader, computeTwiddleFactorsKernel, width / 2, 1);
    }
    
    private void DispatchCompute(ComputeShader shader, int kernelIndex, int width, int height)
    {
        uint x, y, z;
        shader.GetKernelThreadGroupSizes(kernelIndex, out x, out y, out z);
        
        int groupsX = Mathf.CeilToInt(width / (float)x);
        int groupsY = Mathf.CeilToInt(height / (float)y);
        
        shader.Dispatch(kernelIndex, groupsX, groupsY, 1);
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
    
    private void ConvertComplexBufferToTexture(ComputeBuffer complexBuffer, RenderTexture outputTexture)
    {
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetBuffer(convertComplexMagToTexKernel, "Src", complexBuffer);
        fftComputeShader.SetTexture(convertComplexMagToTexKernel, "DstTex", outputTexture);
        DispatchCompute(fftComputeShader, convertComplexMagToTexKernel, width, height);
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
            ShowDebugVisualization(source, destination);
            return;
        }
        
        // Debug pyramid level
        if (showPyramidLevel && useSteerablePyramid)
        {
            ShowPyramidLevel(source, destination);
            return;
        }

        if (applyMotionMagnification)
        {
            if (useSteerablePyramid)
            {
                ProcessFrameWithSteerablePyramid(source, destination);
            }
            else
            {
                ProcessFrameWithMotionMagnification(source, destination);
            }
        }
        else
        {
            Graphics.Blit(source, destination);
            Graphics.Blit(textures["sourceTexture"], textures["previousSourceTexture"]);
        }
    }
    
    private void ShowDebugVisualization(RenderTexture source, RenderTexture destination)
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
        
        Graphics.Blit(textures["sourceTexture"], textures["previousSourceTexture"]);
    }
    
    private void ShowPyramidLevel(RenderTexture source, RenderTexture destination)
    {
        if (pyramidLevelToShow >= 0 && pyramidLevelToShow < currentPyramidLevels.Count)
        {
            // Process frame to get pyramid decomposition
            Graphics.Blit(textures["sourceTexture"], textures["yiqTexture"], rgbToYiqMaterial);
            PadTexture(textures["yiqTexture"], textures["paddedTexture"]);
            ExtractYChannel(textures["paddedTexture"], textures["yChannelTexture"]);
            
            // Perform FFT
            PerformFFT(textures["yChannelTexture"], complexBuffer1, complexBuffer2, textures["currentDFTTexture"]);
            
            // Decompose into pyramid
            DecomposeIntoPyramid(textures["currentDFTTexture"], currentPyramidLevels);
            
            // Show selected level
            CropTexture(currentPyramidLevels[pyramidLevelToShow], destination);
        }
        else
        {
            Graphics.Blit(source, destination);
        }
        
        Graphics.Blit(textures["sourceTexture"], textures["previousSourceTexture"]);
    }
    
    private void ConvertComplexToMagnitude(ComputeBuffer complexBuffer, RenderTexture magnitudeTexture)
    {
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetBuffer(convertComplexMagToTexScaledKernel, "Src", complexBuffer);
        fftComputeShader.SetTexture(convertComplexMagToTexScaledKernel, "DstTex", magnitudeTexture);
        DispatchCompute(fftComputeShader, convertComplexMagToTexScaledKernel, width, height);
    }
    
    private void ConvertComplexToPhase(ComputeBuffer complexBuffer, RenderTexture phaseTexture)
    {
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetBuffer(convertComplexPhaseToTexKernel, "Src", complexBuffer);
        fftComputeShader.SetTexture(convertComplexPhaseToTexKernel, "DstTex", phaseTexture);
        DispatchCompute(fftComputeShader, convertComplexPhaseToTexKernel, width, height);
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
    
    // New method: Process frame with steerable pyramid
    private void ProcessFrameWithSteerablePyramid(RenderTexture source, RenderTexture destination)
    {
        // Convert current frame to YIQ
        Graphics.Blit(textures["sourceTexture"], textures["yiqTexture"], rgbToYiqMaterial);
        PadTexture(textures["yiqTexture"], textures["paddedTexture"]);
        ExtractYChannel(textures["paddedTexture"], textures["yChannelTexture"]);
        
        // Convert previous frame to YIQ
        Graphics.Blit(textures["previousSourceTexture"], textures["previousYiqTexture"], rgbToYiqMaterial);
        PadTexture(textures["previousYiqTexture"], textures["previousPaddedTexture"]);
        ExtractYChannel(textures["previousPaddedTexture"], textures["previousYChannelTexture"]);
        
        // Transform to frequency domain
        PerformFFT(textures["yChannelTexture"], complexBuffer1, complexBuffer2, textures["currentDFTTexture"]);
        PerformFFT(textures["previousYChannelTexture"], previousComplexBuffer1, previousComplexBuffer2, textures["previousDFTTexture"]);
        
        // Decompose into steerable pyramid
        DecomposeIntoPyramid(textures["currentDFTTexture"], currentPyramidLevels);
        DecomposeIntoPyramid(textures["previousDFTTexture"], previousPyramidLevels);
        
        // Extract residuals
        ExtractResiduals(textures["currentDFTTexture"]);
        
        // Process phase differences at each pyramid level
        ProcessPyramidPhaseDifferences();
        
        // Reconstruct from pyramid
        ReconstructFromPyramid(textures["pyramidReconstructionDFT"]);
        
        // Perform IFFT
        PerformIFFT(textures["pyramidReconstructionDFT"], textures["processedYTexture"]);
        
        // Apply anti-aliasing
        ApplyAntiAliasing(textures["processedYTexture"], textures["processedYTexture"]);
        
        // Combine channels
        CombineYIQChannels(textures["processedYTexture"], textures["paddedTexture"], textures["destinationTexture"]);
        
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
        
        // Swap pyramid levels
        var temp = currentPyramidLevels;
        currentPyramidLevels = previousPyramidLevels;
        previousPyramidLevels = temp;
    }
    
    // New method: Decompose image into steerable pyramid
    private void DecomposeIntoPyramid(RenderTexture dftTexture, List<RenderTexture> pyramidLevels)
    {
        if (pyramidDecomposeComputeShader == null) return;
        
        pyramidDecomposeComputeShader.SetInt("_Width", width);
        pyramidDecomposeComputeShader.SetInt("_Height", height);
        pyramidDecomposeComputeShader.SetInt("_NumOrientations", numOrientations);
        pyramidDecomposeComputeShader.SetFloat("_Gain", 1.0f / pyramidLevels);
        
        // Decompose each level (excluding first and last as per PyTorch reference)
        for (int level = 0; level < pyramidLevels; level++)
        {
            pyramidDecomposeComputeShader.SetInt("_ScaleIndex", level);
            pyramidDecomposeComputeShader.SetTexture(decomposeLevelKernel, "_InputDFT", dftTexture);
            pyramidDecomposeComputeShader.SetTexture(decomposeLevelKernel, "_BandPassFilter", pyramidFilters[level]);
            pyramidDecomposeComputeShader.SetTexture(decomposeLevelKernel, "_PyramidLevel", pyramidLevels[level]);
            
            DispatchCompute(pyramidDecomposeComputeShader, decomposeLevelKernel, width, height);
        }
    }
    
    // New method: Extract low-pass and high-pass residuals
    private void ExtractResiduals(RenderTexture dftTexture)
    {
        if (pyramidDecomposeComputeShader == null) return;
        
        pyramidDecomposeComputeShader.SetInt("_Width", width);
        pyramidDecomposeComputeShader.SetInt("_Height", height);
        
        // Extract low-pass residual
        pyramidDecomposeComputeShader.SetTexture(extractResidualLowPassKernel, "_InputDFT", dftTexture);
        pyramidDecomposeComputeShader.SetTexture(extractResidualLowPassKernel, "_LowPassFilter", textures["pyramidLowPassFilter"]);
        pyramidDecomposeComputeShader.SetTexture(extractResidualLowPassKernel, "_OutputDFT", textures["lowPassResidualDFT"]);
        DispatchCompute(pyramidDecomposeComputeShader, extractResidualLowPassKernel, width, height);
        
        // Extract high-pass residual
        pyramidDecomposeComputeShader.SetTexture(extractResidualHighPassKernel, "_InputDFT", dftTexture);
        pyramidDecomposeComputeShader.SetTexture(extractResidualHighPassKernel, "_HighPassFilter", textures["pyramidHighPassFilter"]);
        pyramidDecomposeComputeShader.SetTexture(extractResidualHighPassKernel, "_OutputDFT", textures["highPassResidualDFT"]);
        DispatchCompute(pyramidDecomposeComputeShader, extractResidualHighPassKernel, width, height);
    }
    
    // New method: Process phase differences for each pyramid level
    private void ProcessPyramidPhaseDifferences()
    {
        if (pyramidReconstructComputeShader == null) return;
        
        pyramidReconstructComputeShader.SetInt("_Width", width);
        pyramidReconstructComputeShader.SetInt("_Height", height);
        pyramidReconstructComputeShader.SetInt("_NumOrientations", numOrientations);
        pyramidReconstructComputeShader.SetFloat("_PhaseScale", phaseScale);
        pyramidReconstructComputeShader.SetFloat("_MagnitudeThreshold", magnitudeThreshold);
        
        // Process each pyramid level (excluding first and last as per reference)
        for (int level = 1; level < pyramidLevels - 1; level++)
        {
            pyramidReconstructComputeShader.SetInt("_ScaleIndex", level);
            pyramidReconstructComputeShader.SetTexture(processPhaseDifferencePerLevelKernel, "_CurrentPyramidLevel", currentPyramidLevels[level]);
            pyramidReconstructComputeShader.SetTexture(processPhaseDifferencePerLevelKernel, "_PreviousPyramidLevel", previousPyramidLevels[level]);
            pyramidReconstructComputeShader.SetTexture(processPhaseDifferencePerLevelKernel, "_BandPassFilter", pyramidFilters[level]);
            pyramidReconstructComputeShader.SetTexture(processPhaseDifferencePerLevelKernel, "_ProcessedPyramidLevel", processedPyramidLevels[level]);
            
            DispatchCompute(pyramidReconstructComputeShader, processPhaseDifferencePerLevelKernel, width, height);
        }
        
        // Copy unprocessed levels
        Graphics.Blit(currentPyramidLevels[0], processedPyramidLevels[0]);
        if (pyramidLevels > 1)
        {
            Graphics.Blit(currentPyramidLevels[pyramidLevels - 1], processedPyramidLevels[pyramidLevels - 1]);
        }
    }
    
    // New method: Reconstruct from pyramid
    private void ReconstructFromPyramid(RenderTexture outputDFT)
    {
        if (pyramidReconstructComputeShader == null) return;
        
        pyramidReconstructComputeShader.SetInt("_Width", width);
        pyramidReconstructComputeShader.SetInt("_Height", height);
        pyramidReconstructComputeShader.SetInt("_NumOrientations", numOrientations);
        pyramidReconstructComputeShader.SetInt("_AddHighPass", includeHighPassInReconstruction ? 1 : 0);
        
        // Initialize reconstruction buffer
        pyramidReconstructComputeShader.SetTexture(initializeReconstructionKernel, "_ReconstructionDFT", outputDFT);
        DispatchCompute(pyramidReconstructComputeShader, initializeReconstructionKernel, width, height);
        
        // Accumulate pyramid levels
        for (int level = 0; level < pyramidLevels; level++)
        {
            pyramidReconstructComputeShader.SetInt("_ScaleIndex", level);
            pyramidReconstructComputeShader.SetTexture(accumulatePyramidLevelKernel, "_ProcessedPyramidLevel", processedPyramidLevels[level]);
            pyramidReconstructComputeShader.SetTexture(accumulatePyramidLevelKernel, "_BandPassFilter", pyramidFilters[level]);
            pyramidReconstructComputeShader.SetTexture(accumulatePyramidLevelKernel, "_ReconstructionDFT", outputDFT);
            
            DispatchCompute(pyramidReconstructComputeShader, accumulatePyramidLevelKernel, width, height);
        }
        
        // Add residuals
        pyramidReconstructComputeShader.SetTexture(addResidualsKernel, "_LowPassResidualDFT", textures["lowPassResidualDFT"]);
        pyramidReconstructComputeShader.SetTexture(addResidualsKernel, "_HighPassResidualDFT", textures["highPassResidualDFT"]);
        pyramidReconstructComputeShader.SetTexture(addResidualsKernel, "_LowPassFilter", textures["pyramidLowPassFilter"]);
        pyramidReconstructComputeShader.SetTexture(addResidualsKernel, "_HighPassFilter", textures["pyramidHighPassFilter"]);
        pyramidReconstructComputeShader.SetTexture(addResidualsKernel, "_ReconstructionDFT", outputDFT);
        
        DispatchCompute(pyramidReconstructComputeShader, addResidualsKernel, width, height);
    }
    
    // Updated method - now passes bandpass filter parameters to phase difference shader
    private void ProcessPhaseDifferenceWithComputeShader(RenderTexture currentDFT, RenderTexture previousDFT, RenderTexture outputDFT)
    {
        if (phaseDifferenceComputeShader == null)
        {
            Debug.LogError("Phase Difference Compute Shader is not assigned!");
            Graphics.Blit(currentDFT, outputDFT);
            return;
        }
        
        // Set input/output textures
        phaseDifferenceComputeShader.SetTexture(processPhaseDifferenceKernel, "_CurrentDFT", currentDFT);
        phaseDifferenceComputeShader.SetTexture(processPhaseDifferenceKernel, "_PreviousDFT", previousDFT);
        phaseDifferenceComputeShader.SetTexture(processPhaseDifferenceKernel, "_OutputDFT", outputDFT);
        
        // Phase difference parameters
        phaseDifferenceComputeShader.SetFloat("_PhaseScale", phaseScale);
        phaseDifferenceComputeShader.SetFloat("_MagnitudeThreshold", magnitudeThreshold);
        phaseDifferenceComputeShader.SetFloat("_MagnitudeScale", magnitudeScale);
        
        // Dimensions
        phaseDifferenceComputeShader.SetInt("_Width", width);
        phaseDifferenceComputeShader.SetInt("_Height", height);
        
        // Bandpass filter parameters - now applied to phase delta
        phaseDifferenceComputeShader.SetInt("_ApplyBandpassFilter", applyBandpassFilter ? 1 : 0);
        phaseDifferenceComputeShader.SetFloat("_LowFreqCutoff", lowFrequencyCutoff);
        phaseDifferenceComputeShader.SetFloat("_HighFreqCutoff", highFrequencyCutoff);
        phaseDifferenceComputeShader.SetFloat("_FilterSteepness", filterSteepness);
        
        // Motion enhancement parameters
        phaseDifferenceComputeShader.SetFloat("_MotionSensitivity", motionSensitivity);
        phaseDifferenceComputeShader.SetFloat("_EdgeEnhancement", enhanceEdges ? edgeEnhancement : 0.0f);
        
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
        
        // Convert previous frame to YIQ
        Graphics.Blit(textures["previousSourceTexture"], textures["previousYiqTexture"], rgbToYiqMaterial);
        PadTexture(textures["previousYiqTexture"], textures["previousPaddedTexture"]);
        ExtractYChannel(textures["previousPaddedTexture"], textures["previousYChannelTexture"]);
        
        // Process with FFT (NO BANDPASS FILTER HERE - it's now in phase difference)
        PerformFFT(textures["yChannelTexture"], complexBuffer1, complexBuffer2, textures["currentDFTTexture"]);
        PerformFFT(textures["previousYChannelTexture"], previousComplexBuffer1, previousComplexBuffer2, textures["previousDFTTexture"]);
        
        // Use compute shader for phase difference processing (with bandpass on phase delta)
        ProcessPhaseDifferenceWithComputeShader(textures["currentDFTTexture"], textures["previousDFTTexture"], textures["modifiedDFTTexture"]);
        
        // Perform IFFT on the modified DFT texture
        PerformIFFT(textures["modifiedDFTTexture"], textures["processedYTexture"]);
        
        // Apply anti-aliasing
        ApplyAntiAliasing(textures["processedYTexture"], textures["processedYTexture"]);
        
        // Combine channels
        CombineYIQChannels(textures["processedYTexture"], textures["paddedTexture"], textures["destinationTexture"]);
        
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

    // FFT Implementation - BANDPASS FILTER REMOVED
    private void PerformFFT(RenderTexture yTexture, ComputeBuffer outputBuffer1, ComputeBuffer outputBuffer2, RenderTexture outputTexture)
    {
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);

        fftComputeShader.SetTexture(convertTexToComplexKernel, "SrcTex", yTexture);
        fftComputeShader.SetBuffer(convertTexToComplexKernel, "Dst", outputBuffer1);
        DispatchCompute(fftComputeShader, convertTexToComplexKernel, width, height);

        fftComputeShader.SetBuffer(centerComplexKernel, "Src", outputBuffer1);
        fftComputeShader.SetBuffer(centerComplexKernel, "Dst", outputBuffer2);
        DispatchCompute(fftComputeShader, centerComplexKernel, width, height);

        fftComputeShader.SetBuffer(bitRevByRowKernel, "Src", outputBuffer2);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "Dst", outputBuffer1);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(fftComputeShader, bitRevByRowKernel, width, height);

        ComputeBuffer src = outputBuffer1;
        ComputeBuffer dst = outputBuffer2;
        
        for (int stride = 2; stride <= width; stride *= 2)
        {
            fftComputeShader.SetInt("BUTTERFLY_STRIDE", stride);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "Src", src);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "Dst", dst);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "TwiddleFactors", twiddleFactorsBuffer);
            DispatchCompute(fftComputeShader, butterflyByRowKernel, width, height);
            
            ComputeBuffer temp = src;
            src = dst;
            dst = temp;
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

        ComputeBuffer temp2 = src;
        src = dst;
        dst = temp2;

        for (int stride = 2; stride <= height; stride *= 2)
        {
            fftComputeShader.SetInt("BUTTERFLY_STRIDE", stride);
            fftComputeShader.SetBuffer(butterflyByColKernel, "Src", src);
            fftComputeShader.SetBuffer(butterflyByColKernel, "Dst", dst);
            fftComputeShader.SetBuffer(butterflyByColKernel, "TwiddleFactors", twiddleFactorsBuffer);
            DispatchCompute(fftComputeShader, butterflyByColKernel, width, height);
            
            ComputeBuffer temp = src;
            src = dst;
            dst = temp;
        }

        fftComputeShader.SetBuffer(convertComplexToTexRGKernel, "Src", src);
        fftComputeShader.SetTexture(convertComplexToTexRGKernel, "DstTex", outputTexture);
        DispatchCompute(fftComputeShader, convertComplexToTexRGKernel, width, height);
    }
    
    private void ConvertTextureToBuffer(RenderTexture dftTexture, ComputeBuffer outputBuffer)
    {
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetTexture(convertTextureToComplexKernel, "SrcTex", dftTexture);
        fftComputeShader.SetBuffer(convertTextureToComplexKernel, "Dst", outputBuffer);
        DispatchCompute(fftComputeShader, convertTextureToComplexKernel, width, height);
    }
    
    private void PerformIFFT(RenderTexture dftTexture, RenderTexture outputTexture)
    {
        ConvertTextureToBuffer(dftTexture, complexBuffer1);
        
        fftComputeShader.SetInt("WIDTH", width);
        fftComputeShader.SetInt("HEIGHT", height);
        fftComputeShader.SetBuffer(conjugateComplexKernel, "Src", complexBuffer1);
        fftComputeShader.SetBuffer(conjugateComplexKernel, "Dst", complexBuffer2);
        DispatchCompute(fftComputeShader, conjugateComplexKernel, width, height);
        
        fftComputeShader.SetInt("N", width);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "Src", complexBuffer2);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "Dst", complexBuffer1);
        fftComputeShader.SetBuffer(bitRevByRowKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(fftComputeShader, bitRevByRowKernel, width, height);

        ComputeBuffer src = complexBuffer1;
        ComputeBuffer dst = complexBuffer2;
        
        for (int stride = 2; stride <= width; stride *= 2)
        {
            fftComputeShader.SetInt("BUTTERFLY_STRIDE", stride);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "Src", src);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "Dst", dst);
            fftComputeShader.SetBuffer(butterflyByRowKernel, "TwiddleFactors", twiddleFactorsBuffer);
            DispatchCompute(fftComputeShader, butterflyByRowKernel, width, height);
            
            ComputeBuffer temp = src;
            src = dst;
            dst = temp;
        }

        fftComputeShader.SetInt("N", height);
        fftComputeShader.SetBuffer(bitRevByColKernel, "Src", src);
        fftComputeShader.SetBuffer(bitRevByColKernel, "Dst", dst);
        fftComputeShader.SetBuffer(bitRevByColKernel, "BitRevIndices", bitRevIndicesBuffer);
        DispatchCompute(fftComputeShader, bitRevByColKernel, width, height);

        ComputeBuffer temp2 = src;
        src = dst;
        dst = temp2;

        for (int stride = 2; stride <= height; stride *= 2)
        {
            fftComputeShader.SetInt("BUTTERFLY_STRIDE", stride);
            fftComputeShader.SetBuffer(butterflyByColKernel, "Src", src);
            fftComputeShader.SetBuffer(butterflyByColKernel, "Dst", dst);
            fftComputeShader.SetBuffer(butterflyByColKernel, "TwiddleFactors", twiddleFactorsBuffer);
            DispatchCompute(fftComputeShader, butterflyByColKernel, width, height);
            
            ComputeBuffer temp = src;
            src = dst;
            dst = temp;
        }

        fftComputeShader.SetBuffer(conjugateComplexKernel, "Src", src);
        fftComputeShader.SetBuffer(conjugateComplexKernel, "Dst", dst);
        DispatchCompute(fftComputeShader, conjugateComplexKernel, width, height);

        ComputeBuffer temp3 = src;
        src = dst;
        dst = temp3;

        fftComputeShader.SetBuffer(divideComplexByDimensionsKernel, "Src", src);
        fftComputeShader.SetBuffer(divideComplexByDimensionsKernel, "Dst", dst);
        DispatchCompute(fftComputeShader, divideComplexByDimensionsKernel, width, height);

        ComputeBuffer temp4 = src;
        src = dst;
        dst = temp4;

        fftComputeShader.SetBuffer(centerComplexKernel, "Src", src);
        fftComputeShader.SetBuffer(centerComplexKernel, "Dst", dst);
        DispatchCompute(fftComputeShader, centerComplexKernel, width, height);

        fftComputeShader.SetBuffer(convertComplexMagToTexKernel, "Src", dst);
        fftComputeShader.SetTexture(convertComplexMagToTexKernel, "DstTex", outputTexture);
        DispatchCompute(fftComputeShader, convertComplexMagToTexKernel, width, height);
    }
}