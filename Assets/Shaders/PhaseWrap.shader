// PhaseWrap.shader
Shader "Custom/PhaseWrap"
{
    Properties
    {
        _MainTex ("Main Texture", 2D) = "white" {}
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

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float wrap_phase(float phase)
            {
                // Wrap phase to [-pi, pi] range
                const float PI = 3.14159265;
                
                // Use fmod for proper wrapping
                phase = fmod(phase + PI, 2.0 * PI);
                if (phase < 0) phase += 2.0 * PI;
                return phase - PI;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.uv);
                
                // Wrap phase stored in red channel
                col.r = wrap_phase(col.r);
                
                return col;
            }
            ENDCG
        }
    }
}