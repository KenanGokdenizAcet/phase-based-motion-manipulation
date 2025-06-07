Shader "Hidden/CombineYIQChannels"
{
    Properties
    {
        _YTex ("Y Channel Texture", 2D) = "white" {}
        _IQTex ("IQ Channels Texture", 2D) = "white" {}
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

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

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }
            
            sampler2D _YTex;     // Contains processed Y channel (in R component)
            sampler2D _IQTex;    // Contains original YIQ data

            fixed4 frag (v2f i) : SV_Target
            {
                // Get processed Y channel from the Y texture (stored in R component)
                float y = tex2D(_YTex, i.uv).r;
                
                // Get original I and Q channels from the YIQ texture
                // In YIQ, I is stored in G component and Q in B component
                float4 origYIQ = tex2D(_IQTex, i.uv);
                float i_val = origYIQ.g;
                float q_val = origYIQ.b;
                
                // Combine into a new YIQ texture
                return fixed4(y, i_val, q_val, 1.0);
            }
            ENDCG
        }
    }
}