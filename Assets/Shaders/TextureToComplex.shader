Shader "Hidden/TextureToComplex"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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
            
            sampler2D _MainTex;

            struct ComplexNumber
            {
                float real;
                float imag;
            };

            ComplexNumber frag (v2f i) : SV_Target
            {
                // Read RG channels from texture as complex number (R=real, G=imag)
                float4 col = tex2D(_MainTex, i.uv);
                
                // Output as complex number structure
                ComplexNumber output;
                output.real = col.r;  // Real part
                output.imag = col.g;  // Imaginary part
                
                return output;
            }
            ENDCG
        }
    }
}