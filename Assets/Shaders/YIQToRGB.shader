Shader "Custom/YIQToRGB"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _YMultiplier ("Y Multiplier", Float) = 1.0
        _IMultiplier ("I Multiplier", Float) = 1.0
        _QMultiplier ("Q Multiplier", Float) = 1.0
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
            #pragma multi_compile _ _YIQADJUSTMENT_ON
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

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float _YMultiplier;
            float _IMultiplier;
            float _QMultiplier;
            
            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }
            
            // YIQ to RGB conversion matrix
            // This is the inverse of the RGB to YIQ matrix
            static const float3x3 YIQ_TO_RGB = float3x3(
                1.0, 0.956, 0.621,
                1.0, -0.272, -0.647,
                1.0, -1.106, 1.703
            );

            fixed4 frag (v2f i) : SV_Target
            {
                // Sample the texture (expecting YIQ color values)
                fixed4 inputColor = tex2D(_MainTex, i.uv);
                
                // Apply YIQ adjustments if enabled
                float3 yiqColor = inputColor.rgb;
                
                #ifdef _YIQADJUSTMENT_ON
                    // Apply multipliers to each YIQ component
                    yiqColor.x *= _YMultiplier;   // Y (luminance)
                    yiqColor.y *= _IMultiplier;   // I (orange-blue)
                    yiqColor.z *= _QMultiplier;   // Q (purple-green)
                #endif
                
                // Apply YIQ to RGB transformation
                float3 rgbColor = mul(YIQ_TO_RGB, yiqColor);
                
                // Ensure RGB values stay in valid range [0,1]
                rgbColor = saturate(rgbColor);
                
                // Output result with original alpha
                return fixed4(rgbColor, inputColor.a);
            }
            ENDCG
        }
    }
    FallBack "Diffuse"
} 