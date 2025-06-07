// GaussianBlur.shader
Shader "Hidden/GaussianBlur"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _BlurSize ("Blur Size", Float) = 1.0
        _Direction ("Direction", Vector) = (1,0,0,0)
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
            float4 _MainTex_TexelSize;
            float _BlurSize;
            float4 _Direction;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float2 texelSize = _MainTex_TexelSize.xy * _BlurSize;
                float2 direction = _Direction.xy;
                
                // 5-tap Gaussian blur
                fixed4 col = tex2D(_MainTex, i.uv) * 0.2270270270;
                col += tex2D(_MainTex, i.uv + direction * texelSize * 1.3846153846) * 0.3162162162;
                col += tex2D(_MainTex, i.uv - direction * texelSize * 1.3846153846) * 0.3162162162;
                col += tex2D(_MainTex, i.uv + direction * texelSize * 3.2307692308) * 0.0702702703;
                col += tex2D(_MainTex, i.uv - direction * texelSize * 3.2307692308) * 0.0702702703;
                
                return col;
            }
            ENDCG
        }
    }
}