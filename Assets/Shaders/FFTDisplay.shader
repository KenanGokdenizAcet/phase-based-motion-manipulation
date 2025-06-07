Shader "Custom/FFTDisplay"
{
    Properties
    {
        _SpectrumTexture ("Spectrum Texture", 2D) = "black" {}
        _OutputTexture ("Output Texture", 2D) = "black" {}
        _ShowMagnitude ("Show Magnitude", Int) = 1
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _SpectrumTexture;
            sampler2D _OutputTexture;
            int _ShowMagnitude;
            
            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }
            
            fixed4 frag (v2f i) : SV_Target
            {
                // Determine which half of the screen we're in
                bool isLeftHalf = i.uv.x < 0.5;
                
                // Adjust UV coordinates for aspect ratio and scaling
                float2 uv = i.uv;
                uv.x = isLeftHalf ? uv.x * 2.0 : (uv.x - 0.5) * 2.0;
                
                // Sample appropriate texture
                fixed4 col;
                if (isLeftHalf)
                {
                    // Show spectrum (magnitude or phase)
                    col = tex2D(_SpectrumTexture, uv);
                    
                    // Apply colorization to make spectrum more visible
                    if (_ShowMagnitude == 1)
                    {
                        // Hot color mapping for magnitude
                        float intensity = (col.r + col.g + col.b) / 3.0;
                        col = float4(intensity * 2.0, intensity, intensity * 0.5, 1.0);
                    }
                    else
                    {
                        // Phase visualization (rainbow colors)
                        float phase = atan2(col.g, col.r);
                        float norm_phase = (phase + 3.14159) / (2.0 * 3.14159);
                        
                        // Simple HSV to RGB conversion
                        float h = norm_phase * 6.0;
                        float i = floor(h);
                        float f = h - i;
                        float q = 1.0 - f;
                        
                        if (i == 0.0) col = float4(1.0, f, 0.0, 1.0);
                        else if (i == 1.0) col = float4(q, 1.0, 0.0, 1.0);
                        else if (i == 2.0) col = float4(0.0, 1.0, f, 1.0);
                        else if (i == 3.0) col = float4(0.0, q, 1.0, 1.0);
                        else if (i == 4.0) col = float4(f, 0.0, 1.0, 1.0);
                        else col = float4(1.0, 0.0, q, 1.0);
                    }
                }
                else
                {
                    // Show reconstructed image
                    col = tex2D(_OutputTexture, uv);
                }
                
                // Draw a thin separator line between the two halves
                if (abs(i.uv.x - 0.5) < 0.002)
                {
                    col = fixed4(1, 1, 1, 1);
                }
                
                return col;
            }
            ENDCG
        }
    }
}