Shader "Custom/RGBToYIQ"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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

            sampler2D _MainTex;
            float4 _MainTex_ST;
            
            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }
            
            // RGB to YIQ conversion matrix
            // Y: Luminance (brightness)
            // I: Orange-blue chrominance (hue/saturation)
            // Q: Purple-green chrominance (hue/saturation)
            static const float3x3 RGB_TO_YIQ = float3x3(
                0.299, 0.587, 0.114,
                0.596, -0.274, -0.322,
                0.211, -0.523, 0.312
            );

            fixed4 frag (v2f i) : SV_Target
            {
                // Sample the texture
                fixed4 inputColor = tex2D(_MainTex, i.uv);
                
                // Apply RGB to YIQ transformation to RGB channels
                float3 yiqColor = mul(RGB_TO_YIQ, inputColor.rgb);
                
                // Output result with original alpha
                return fixed4(yiqColor, inputColor.a);
            }
            ENDCG
        }
    }
    FallBack "Diffuse"
} 