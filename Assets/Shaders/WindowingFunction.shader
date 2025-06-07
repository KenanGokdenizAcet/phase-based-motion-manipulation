Shader "Hidden/WindowingFunction"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Width ("Width", Float) = 1024
        _Height ("Height", Float) = 1024
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
            float _Width;
            float _Height;

            // Hanning penceresi fonksiyonu
            float HanningWindow(float x)
            {
                return 0.5 * (1.0 - cos(2.0 * 3.14159265359 * x));
            }

            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.uv);
                
                // Windowing fonksiyonu için normalize edilmiş koordinatlar (0-1 aralığında)
                float nx = i.uv.x;
                float ny = i.uv.y;
                
                // Hanning pencere değerlerini hesapla
                float windowX = HanningWindow(nx);
                float windowY = HanningWindow(ny);
                
                // 2D pencere değeri (X ve Y pencere değerlerinin çarpımı)
                float window = windowX * windowY;
                
                // Pencere değerini görüntüye uygula
                col.rgb *= window;
                
                return col;
            }
            ENDCG
        }
    }
}