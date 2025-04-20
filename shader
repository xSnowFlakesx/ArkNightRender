Shader "Unlit/pbr"
{
    Properties
    {
        _BaseColorTex("Base Color Tex", 2D) = "white"{}
        _Color("Color", Color) = (1,1,1,1)
        _NormalTex("Normal Tex", 2D) = "bump" {}
        _BumpScale("Bump Scale", Range(-5,5)) = 1
        _ILMTex("ILM Tex", 2D) = "white" {}
        _SDFTex("SDF Tex", 2D) = "white" {}
        _RampTex ("Ramp Texture", 2D) = "white" {}
        _RampTex2 ("Ramp Texture2", 2D) = "white" {}
        _ShadowThreshold ("Shadow Threshold", Range(0,1)) = 0.5
        _ShadowSmoothness ("Shadow Smoothness", Range(0,1)) = 0.1
        _RampThreshold ("Ramp Threshold", Range(0,1)) = 0.5
        _MaterialIDUSE("Material ID USE", Range(0,4)) = 0
        _SDFIDUSE("SDF ID USE", Range(0,4)) = 0
        _OcclusionIntensity("Occlusion Intensity", Range(0,1)) = 0
        _MetallicIntensity("Metallic Intensity", Range(0,1)) = 0
        _MetallicColor("Metallic Color",Color) = (0,0,0,1)
        _RoughnessIntensity("Roughness Intensity", Range(0,5)) = 0
        _SmoothnessIntensity("Smoothness Intensity", Range(0,1)) = 0
        _SpecularTintIntensity("Specular Tint Intensity", Range(0,1)) = 0
        _ShadowIntensity("Shadow Intensity", Range(0,1)) = 0
        _ShaodwColor("Shadow Color", Color) = (0,0,0,1)
        _SHIntensity("SH Intensity", Range(0,1)) = 0
        _aoUsage("AO Usage", Range(0,1)) = 0
        _AmbientIntensity("Ambient Intensity", Range(0,1)) = 0
        _Ambientpower("Ambient Power", Range(0,1)) = 0
        _EmissionMask("Emission Mask", 2D) = "white" {}
        _FresnelPower("Fresnel Power", Range(0, 10)) = 5.0   // 控制边缘宽度
        _FresnelScale("Fresnel Scale", Range(0, 1)) = 0.5    // 控制强度
        _EdgeColor("Edge Color", Color) = (1, 0.5, 0, 1)     // 边缘光颜色

         _ThicknessMap("Thickness Map", 2D) = "white" {}
        _ThicknessColor("Thickness Color", Color) = (0.8,0.5,0.4,1) // 模拟皮下血管颜色
        _ThicknessPower("Thickness Power", Range(0,5)) = 2.0
        _DepthContrast("Depth Contrast", Range(0,5)) = 1.5

        _GGXHair("GGX Hair", Range(0,1)) = 0
        _Anisotropy("Anisotropy", Range(-1, 1)) = 0
        _Specular("Specular", Range(0,1)) = 0.5
        _specularGGXintensity("GGX Specular Intensity", Range(0,100)) = 50
        _AnisoDirectionMap("Anisotropy Direction Map", 2D) = "white" {}
        _AnisoDirectionMap_ST("AnisoDirectionMap_ST", Range(-50,50)) = 0
        _MetallicIntensityGGX("GGX Metallic Intensity", Range(0,1)) = 0

        _SpecularPowerValue("_Specular Power Value",Range(0,50)) = 10
        _SpecularScaleValue("_SpecularS cale Value",Range(0,10)) = 1
        _SpecularphongIntensity("Specular phong Intensity",Range(0,50)) = 5
        _SpecularphongIntensitytint("Specular phong Intensitytint",Color) = (0,0,0,1)



        [HideInInspector]_HeadCenter("Head Center", Vector) = (0,0,0)
        [HideInInspector]_HeadForward("Head Forward", Vector) = (0,0,0)
        [HideInInspector]_HeadRight("Head Right", Vector) = (0,0,0)

        _FaceshadowOffset("Face shadow Offset", Range(0,1)) = 0.1
        _TransitionWidth("Transition Width", Range(0,1)) = 0.05
        //_EmissionIntensity("Emission Intensity", Range(0,1)) = 0

        //_Roughness("Roughness", Range(0,1)) = 0
        //_Metallic("Metallic", Range(0,1)) = 0
        //_Smoothness("Smoothness", Range(0,1)) = 0.5
        _SpecularTint("Specular Tint", Color) = (1,1,1,1)

        _EmissionColor("Emission color", Color) = (0,0,0,0)
        _EmissionIntensity("Emission Intensity", Range(0,10)) = 1
        //_EmissionIntensity("Emission Intensity", Range(0,1)) = 0
        _AlphaClip("Alpha Clip", Range(0,1)) = 0.333
        _OutLineWidth("Outline Width", Range(0,1)) = 0.01
        _MaxOutlineZoffset("Max Outline Zoffset", Range(0,1)) = 0.01
        _OutlineColor("Outline Color", Color) = (0,0,0,1)


        [Header(Option)]
        [Enum(UnityEngine.Rendering.CullMode)] _Cull ("Cull(Default back)", Float) = 2
        [Enum(Off,0,On,1)] _Zwrite("Zwrite (Default On)", Float) = 1
        [Enum(UnityEngine.Rendering.BlendMode)] _SrcBlendMode("SrcBlendMode(Default One)", Float) = 1
        [Enum(UnityEngine.Rendering.BlendMode)] _DstBlendMode("DstBlendMode(Default Zero)", Float) = 0
        [Enum(UnityEngine.Rendering.BlendOp)] _BlendOp("BlendOp(Default Add)", Float) = 0
        _StencilRef("Stencil reference", int) = 0
        [Enum(UnityEngine.Rendering.CompareFunction)] _StencilComp("Stencil comparison function", int) = 0
        [Enum(UnityEngine.Rendering.StencilOp)] _StencilPassOp("Stencil pass operation", int) = 0
        [Enum(UnityEngine.Rendering.StencilOp)] _StencilFailOp("Stencil fail operation", int) = 0
        [Enum(UnityEngine.Rendering.StencilOp)] _StencilZFailOp("Stencil Z fail operation", int) = 0

        [Header(SRP Default)]
        [Toggle(_SRP_DEFAULT_PASS)] _SRPDefaultPass("SRP Default Pass", int) = 0
        [Enum(UnityEngine.Rendering.BlendMode)] _SRPSrcBlendMode("SRP SrcBlendMode(Default One)", Float) = 1
        [Enum(UnityEngine.Rendering.BlendMode)] _SRPDstBlendMode("SRP DstBlendMode(Default Zero)", Float) = 0
        [Enum(UnityEngine.Rendering.BlendOp)] _SRPBlendOp("SRP BlendOp(Default Add)", Float) = 0
        _SRPStencilRef("SRP Stencil reference", int) = 0
        [Enum(UnityEngine.Rendering.CompareFunction)] _SRPStencilComp("SRP Stencil comparison function", int) = 0
        [Enum(UnityEngine.Rendering.StencilOp)] _SRPStencilPassOp("SRP Stencil pass operation", int) = 0
        [Enum(UnityEngine.Rendering.StencilOp)] _SRPStencilFailOp("SRP Stencil fail operation", int) = 0
        [Enum(UnityEngine.Rendering.StencilOp)] _SRPStencilZFailOp("SRP Stencil Z fail operation", int) = 0
    }

    
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }
        LOD 300

        
        HLSLINCLUDE
        
        #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
        #pragma multi_compile _ _MAIN_LIGHT_SHADOWS_CASCADE
        #pragma multi_compile _ _MAIN_LIGHT_SHADOWS_SCREEN

        #pragma multi_compile_fragment _ _LIGHT_LAYERS
        #pragma multi_compile_fragment _ _LIGHT_COOKIES
        #pragma multi_compile_fragment _ _SCREEN_SPACE_OCCLUSION
        #pragma multi_compile_fragment _ _SHADOWS_SOFT
        #pragma multi_compile_fragment _ _ADDITIONAL_LIGHTS_SHADOWS
        #pragma multi_compile_fragment _ _REFLECTION_PROBE_BLENDING
        #pragma multi_compile_fragment _ _REFLECTION_PROBE_BOX_PROJECTION

               
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/BRDF.hlsl"
        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/BSDF.hlsl"
        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/CommonMaterial.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Deprecated.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/SurfaceData.hlsl"

        //#define PI 3.141592654
        #define Eplison 0.001
        #define EPSILON 1e-5

        #define kDielectricSpec half4(0.04, 0.04, 0.04, 1.0 - 0.04) // standard dielectric reflectivity coef at incident angle (= 4%)
        
        float3 CalculateAmbientLighting(
        BRDFData brdfData, 
        float3 normalWS, 
        float3 viewDirectionWS, 
        float occlusion
    ) {
        // 1. 漫反射环境光（球谐函数）
        float3 indirectDiffuse = SampleSH(normalWS) * brdfData.diffuse * occlusion;

        // 2. 镜面反射环境光（基于反射探针或环境贴图）
        float3 reflectVector = reflect(-viewDirectionWS, normalWS);
        float3 indirectSpecular = GlossyEnvironmentReflection(
            reflectVector,
            brdfData.perceptualRoughness,
            occlusion
        );

        //D
            

    return indirectDiffuse + indirectSpecular;
}
// Schlick-GGX几何遮蔽
        float G_SchlickGGX(float NoV, float k)
        {
            return NoV / (NoV * (1.0 - k) + k);
        }

// Smith联合几何遮蔽
        float G_Smith(float NoV, float NoL, float Roughness)
        {
            float k = (Roughness + 1.0) * (Roughness + 1.0) / 8.0;
            return G_SchlickGGX(NoL, k) * G_SchlickGGX(NoV, k);
        }

        // Schlick菲涅尔近似
        float3 F_Schlick(float cosTheta, float3 F0)
        {
            return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
        }

        // 能量守恒版本菲涅尔
        float3 F_SchlickRoughness(float cosTheta, float3 F0, float Roughness)
        {
            return F0 + (max(1.0 - Roughness, F0) - F0) * pow(1.0 - cosTheta, 5.0);
        }


        float AnisoGGXDistribution(float NoH, float HdotX, float HdotY, float Roughness, float anisotropy) {
            float aspect = sqrt(1.0 - anisotropy * 0.9);
            float ax = Roughness * Roughness / aspect;
            float ay = Roughness * Roughness * aspect;
            return 1.0 / (PI * ax * ay * pow((HdotX*HdotX/(ax*ax) + HdotY*HdotY/(ay*ay) + NoH*NoH), 2.0));
        }



        // 各向异性GGX分布函数
            float D_GGXAniso(float ax, float ay, float NoH, float3 H, float3 X, float3 Y)
            {
                float XoH = dot(X, H);
                float YoH = dot(Y, H);
                float d = XoH * XoH / (ax * ax) + YoH * YoH / (ay * ay) + NoH * NoH;
                return 1.0 / (PI * ax * ay * d * d);
            }

            // Smith各向异性可见性项
            float V_SmithGGXCorrelatedAniso(float at, float ab, float ToV, float BoV, float ToL, float BoL, float NoV, float NoL)
            {
                float lambdaV = NoL * length(float3(at * ToV, ab * BoV, NoV));
                float lambdaL = NoV * length(float3(at * ToL, ab * BoL, NoL));
                return 0.5 / (lambdaV + lambdaL);
            }

        

        CBUFFER_START(UnityPerMaterial)

            //sampler2D _BaseColorTex;
            float4 _BaseColorTex_ST;
            float4 _Color;
            TEXTURE2D(_NormalTex);
            SAMPLER(sampler_NormalTex);

            sampler2D _SDFTex;
            //SAMPLER(sampler_SDFTex);

            TEXTURE2D(_ThicknessMap);
            SAMPLER(sampler_ThicknessMap);

            half4 _ThicknessColor;
                float _ThicknessPower;
                float _DepthContrast;
            
            TEXTURE2D(_BaseColorTex);
            //TEXTURE2D(_NormalTex);
            //TEXTURE2D(_OcclusionTex);
            TEXTURE2D(_ILMTex);
            SAMPLER(sampler_ILMTex);
            TEXTURE2D( _RampTex);
            SAMPLER(sampler_RampTex);
            TEXTURE2D( _RampTex2);
            SAMPLER(sampler_RampTex2);
            sampler2D _EmissionMask;
            float4 _MetallicColor;
            //float4 _Color;
            float _RampThreshold;
            float _ShadowThreshold;
            float _ShadowSmoothness;
            float _ShadowIntensity;
            float4 _ShaodwColor;
            float _SHIntensity;
            float _aoUsage;
            float _AmbientIntensity;
            float _Ambientpower;
            float3 _HeadCenter;
            float3 _HeadForward;
            float3 _HeadRight;
            float _FaceshadowOffset;
            float _TransitionWidth;
            float _FresnelPower;
            float _FresnelScale;
            float4 _EdgeColor;
            float _SpecularPowerValue;
            float _SpecularScaleValue;
            float _SpecularphongIntensity;
            float3 _SpecularphongIntensitytint;
            
            float GGXHair;
            float _Anisotropy;
            sampler2D _AnisoDirectionMap;
            float _specularGGXintensity;
            float _AnisoDirectionMap_ST;
            float _Specular;
            float _MetallicIntensityGGX;

            
            //SAMPLER(sampler_LinearRepeat);
            
            //sampler2D _OcclusionTex;
            float _BumpScale;
            float _OcclusionIntensity;
            float _MetallicIntensity;
            float _RoughnessIntensity;
            float _SmoothnessIntensity;
            float _SpecularTintIntensity;
            //float4 _OcclusionTex_ST;
            //float4 _ILMTex;
            //float _Roughness;
            //float _Smoothness;
            float4 _SpecularTint;
            //float _Metallic;
            float4 _EmissionColor;
            float _EmissionIntensity;
            float _AlphaClip;
            float _MaterialIDUSE;
            float _SDFIDUSE;

        CBUFFER_END

        


        //SAMPLER(sampler_LinearRepeat);

        struct UniversalAttributes
        {
            float4 positionOS : POSITION;
            float3 normalOS : NORMAL;
            float4 tangentOS : TANGENT;
            float2 texcoord : TEXCOORD0;
        };

        struct UniversalVaryings
        {
            float2 uv                      : TEXCOORD0;
            float4 positionWSAndFogFactor  : TEXCOORD1;
            float3 normalWS                : TEXCOORD2;
            float4 tangentWS               : TEXCOORD3;
            float3 bitangentWS             : TEXCOORD4;
            float3 viewDirWS                : TEXCOORD5;
            float4 positionCS              : SV_POSITION;
            float3 SH                      : TEXCOORD6;  
            float3 positionWS : TEXCOORD7;       
            float4 shadowCoord : TEXCOORD8; // 新增阴影坐标     
        };

        UniversalVaryings MainVS(UniversalAttributes input)
        {
            VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);
            VertexNormalInputs normalInputs = GetVertexNormalInputs(input.normalOS, input.tangentOS);


            UniversalVaryings output;
            output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
            output.positionWS = TransformObjectToWorld(input.positionOS.xyz);
            //output.positionCS = positionInputs.positionCS;
            output.positionWSAndFogFactor = float4(positionInputs.positionWS,ComputeFogFactor(positionInputs.positionCS.z));

            // 计算世界空间法线
    output.normalWS = TransformObjectToWorldNormal(input.normalOS);
    
    // 计算世界空间切线
    output.tangentWS.xyz = TransformObjectToWorldDir(input.tangentOS.xyz);
    output.tangentWS.w = input.tangentOS.w; // 保存手性
    
    // 计算副切线（叉乘法线与切线）
    output.bitangentWS = cross(output.normalWS, output.tangentWS.xyz) * 
                        (input.tangentOS.w * unity_WorldTransformParams.w);
            output.viewDirWS = unity_OrthoParams.w == 0 ? GetCameraPositionWS() - positionInputs.positionWS : GetWorldToViewMatrix()[2].xyz;
            output.uv = input.texcoord;
            output.SH = SampleSH(lerp(normalInputs.normalWS,float3(0,0,0),_SHIntensity));
            output.shadowCoord = TransformWorldToShadowCoord(output.positionWS);
            return output;
        }

        float4 MainPS(UniversalVaryings input) : SV_Target
        {
            
            float4 mainTex = SAMPLE_TEXTURE2D(_BaseColorTex, sampler_LinearRepeat, input.uv);
            float3 baseColor = mainTex.rgb * _Color.rgb;
            float baseAlpha = 1.0;

            
            float3 normalWS = normalize(input.normalWS); 
            
            float3 pixelNormalWS = normalWS;
            if (_MaterialIDUSE == 0)
                {
                    // 使用自身法线
                    pixelNormalWS = normalWS;
                }
                else
                {
                    // 使用法线贴图

                // float4 lightData = SAMPLE_TEXTURE2D(_NormalTex,sampler_NormalTex, input.uv);
                // lightData = lightData * 2 - 1;
                // //diffuseBias = lightData.z * 2;

                // float sgn = input.tangentWS.w;
                // float3 tangentWS = normalize(input.tangentWS.xyz);
                // float3 bitangentWS = cross(normalWS, tangentWS) * sgn;

                // float3 pixelNormalTS = float3(lightData.xy,0.0);
                // pixelNormalTS.xy *= _BumpScale;
                // pixelNormalTS.z = sqrt(1.0 - min(0.0, dot(pixelNormalTS.xy, pixelNormalTS.xy)));
                // pixelNormalWS = TransformTangentToWorld(pixelNormalTS,float3x3(tangentWS,bitangentWS,normalWS));
                // pixelNormalWS = normalize(pixelNormalWS);

                // 采样法线贴图
    half4 packedNormal = SAMPLE_TEXTURE2D(_NormalTex, sampler_NormalTex, input.uv);
    
    // 解包法线（从[0,1]映射到[-1,1]）
    half3 pixelNormalTS = UnpackNormalScale(packedNormal, _BumpScale);
    
    // 构建TBN矩阵
    float3x3 TBN = float3x3(
        input.tangentWS.xyz,
        input.bitangentWS,
        input.normalWS
    );
    
    // 将法线转换到世界空间
     pixelNormalWS = normalize(mul(pixelNormalTS, TBN));
            }

                //float3 baseColor = mainTex.rgb * _Color.rgb;
                 // 3. 定义 SurfaceData
                 SurfaceData surfaceData = (SurfaceData)0;
                surfaceData.albedo = baseColor;
                surfaceData.metallic = SAMPLE_TEXTURE2D(_ILMTex, sampler_ILMTex, input.uv).x * _MetallicIntensity;
                surfaceData.specular = kDielectricSpec.rgb;
                surfaceData.smoothness =  1 - SAMPLE_TEXTURE2D(_ILMTex, sampler_ILMTex, input.uv).y * _RoughnessIntensity;
                surfaceData.occlusion = SAMPLE_TEXTURE2D(_ILMTex, sampler_ILMTex, input.uv).z * _OcclusionIntensity;
                surfaceData.alpha = baseAlpha;
                surfaceData.emission = _EmissionColor.rgb * _EmissionIntensity;
                surfaceData.clearCoatMask = 0.0;
                surfaceData.clearCoatSmoothness = 0.0;

                // URP 12+ 兼容
                #if defined(UNIVERSAL_PIPELINE_12_OR_NEWER)
                    surfaceData.diffuseAlpha = 1.0;
                #endif

                // URP 14+ 兼容
                #if defined(UNIVERSAL_PIPELINE_14_OR_NEWER)
                    surfaceData.geomNormalWS = input.normalWS;
                #endif

                BRDFData brdfData; // ✅ 正确声明
                InitializeBRDFData(surfaceData, brdfData);

                //BRDFData brdf = DirectBDRF_Aniso(pbrInput, surfaceData, light, anisotropy);


            float3 positionWS = input.positionWSAndFogFactor.xyz;

            float4 shadowCoord = TransformWorldToShadowCoord(positionWS);
            Light mainLight = GetMainLight();
            float3 lightDirWS = normalize(mainLight.direction);
            float3 lightDirectionWSFloat3 = lightDirWS;
            float3 halfDirWS = SafeNormalize(lightDirectionWSFloat3 + input.viewDirWS);
            float3 V = normalize(input.viewDirWS);
            float3 N = pixelNormalWS;
            float3 L = lightDirectionWSFloat3;
            float3 H = halfDirWS;

            float3 T = normalize(input.tangentWS);
            float3 B = normalize(input.bitangentWS);

            float NoL = saturate(dot(N, L));
            float NoV = saturate(dot(N, V));
            float NoH = saturate(dot(N, H));
            float VoH = saturate(dot(V, H));

            float ToH = dot(T, H);
                float BoH = dot(B, H);
                float ToV = dot(T, V);
                float BoV = dot(B, V);
                float ToL = dot(T, L);
                float BoL = dot(B, L);


            float materialid = SAMPLE_TEXTURE2D(_ILMTex, sampler_ILMTex, input.uv).y;
               

            float Roughness = SAMPLE_TEXTURE2D(_ILMTex, sampler_ILMTex, input.uv).a * _RoughnessIntensity;
             Roughness = max(Roughness, 0.001);
             float Metallic = SAMPLE_TEXTURE2D(_ILMTex, sampler_ILMTex, input.uv).x * _MetallicIntensity;

             

             float3 F0 = lerp(float3(0.04,0.04,0.04), baseColor, Metallic);

            // BRDF计算 ------------------------------
                // 法线分布
                float D = D_GGX(NoH, Roughness);
                // 几何遮蔽
                float G = G_Smith(NoV, NoL, Roughness);
                
                // 菲涅尔项
                float3 F = F_Schlick(VoH, F0);


                // 镜面反射项
                float3 specular = (D * G * F) / max(4.0 * NoV * NoL, EPSILON);
                float3 specularmetllic = specular * Metallic;

                // 漫反射项
                float3 kD = max((1.0 - F),0.1) * (1.0 - Metallic);
                float3 diffusePBR = kD * baseColor / PI;

                // 光照组合
                float3 radiance = mainLight.color * mainLight.distanceAttenuation;
                float3 directLight = (diffusePBR + specular) * radiance * NoL;

                // 环境光照 ------------------------------
                // 漫反射环境
                float3 ambientDiffuse = SampleSH(N) * baseColor * kD;
                
                // 镜面反射环境
                float3 R = reflect(-V, N);
                float3 prefilteredColor = GlossyEnvironmentReflection(
                    R,
                    Roughness * Roughness,
                    1.0
                );
                float3 ambientSpecular = prefilteredColor * F_SchlickRoughness(NoV, F0, Roughness);
                
                float3 ambientPBR = (ambientDiffuse + ambientSpecular) * 1;

                 // 最终颜色
                float3 finalColorPBR = directLight + ambientPBR;


            
            //float NoH = saturate(dot(pixelNormalWS, halfDir));
            //half LoH = half(saturate(dot(lightDirectionWSFloat3, halfDir)));
            

            float3 lightColor = mainLight.color;
            float3 diffuse = 0;


             


            float ao = surfaceData.occlusion;
         
            float3 ambient = input.SH.rgb * _Ambientpower;
            ambient *= lerp(1, ao, _aoUsage);
            ambient = lerp(ambient, baseColor, _AmbientIntensity);
            //float3 Radiance = mainLight.color;


            // float HV = max(dot(halfDirWS, V), 0);
            //     float NV = max(dot(N, V), 0);
            //     float NL = max(dot(N, L), 0);

                
                

                

                //float3 KS = F;
                // float3 KD = 1 - KS;
                // KD *= 1 - Metallic;
                // float3 nominator = D * F * G;
                // float denominator = max(4 * NV * NL, 0.001);
                // float3 Specular = nominator / denominator;
                // // Specular =max( Specular,0);

                // //Diffuse
                // // float3 Diffuse = KD * BaseColor / PI;
                // float3 Diffuse = KD * baseColor; //没有除以 PI

                // float3 DirectLight = (Diffuse + Specular) * NL * Radiance;

            float EmissionMask = tex2D(_EmissionMask, input.uv);
            float3 EmissionColor = _EmissionColor.rgb * _EmissionIntensity * EmissionMask;

            // 菲涅尔计算
                //float3 normalWS = normalize(input.normalWS);
                //float3 viewDirWS = normalize(input.viewDirWS);
                float fresnel = pow(1.0 - saturate(dot(pixelNormalWS, normalize(input.viewDirWS))), _FresnelPower);
                fresnel = smoothstep(0.3,0.8,fresnel);
                float3 edgeGlow = mainLight.color * _EdgeColor.rgb * fresnel * _FresnelScale * Roughness;
                float3 edgeGlow2 = mainLight.color * _EdgeColor.rgb * fresnel * _FresnelScale;
                edgeGlow += edgeGlow2;
                edgeGlow *= lerp(baseColor,float3(1,1,1),0.9);
                edgeGlow *= NoL;

                float4 Specularphong = pow(NoH,_SpecularPowerValue)*_SpecularScaleValue * Metallic;
                float SpecularphongIntensity = _SpecularphongIntensity;
                float3 SpecularphongIntensitytint = _SpecularphongIntensitytint;
                Specularphong = smoothstep(0.2,0.7,Specularphong);
                float3 SpecularphongColor = Specularphong * baseColor *  SpecularphongIntensitytint * SpecularphongIntensity;

//          Cloth
            if (_SDFIDUSE == 0)
            {
            float  NoL = saturate(dot(pixelNormalWS, lightDirWS));
            float stepNoL = smoothstep(0.0, 1, NoL);
            float HalfLambert = saturate(0.5 + 0.5 * stepNoL);

            
             
            //HalfLambert = pow(HalfLambert, 2.0);


            //float3 ambient = CalculateAmbientLighting(brdfData, pixelNormalWS, input.viewDirWS, surfaceData.occlusion);
            //float HalfLambert = saturate(0.5 + 0.5 * stepNoL);


            float shadowMask = smoothstep(
            _ShadowThreshold - _ShadowSmoothness,
            _ShadowThreshold + _ShadowSmoothness,
            HalfLambert
        );

        //float  HalfLambertStep = smoothstep(0.2, 0.9, HalfLambert);
            // 
            float rampU = HalfLambert;
            float rampV = _RampThreshold;
            float3 rampColor = SAMPLE_TEXTURE2D(_RampTex, sampler_LinearClamp, float2(rampU, rampV)).rgb;

            float rampU2 = HalfLambert;
            //float2 rampUV2 = float2(step(HalfLambert , (1 - shadowMask)), 0.5);
            float rampV2 = _RampThreshold;
            float3 rampColor2 = SAMPLE_TEXTURE2D(_RampTex2, sampler_LinearClamp, float2(rampU, rampV)).rgb;

            float shadow = 1 - smoothstep(
    (1 - shadowMask) - _ShadowSmoothness, 
    (1 - shadowMask) + _ShadowSmoothness, 
    HalfLambert
);
            shadow = smoothstep(0.1, 1.0, shadow);
            
            rampColor2 = lerp(rampColor2, float3(1,1,1), 0.7);
            rampColor2 *= shadow * 0.6;
            
            //rampColor2 = step(rampColor2,shadowMask);


            //half3 shadowColor = lerp(baseColor, baseColor * rampColor2, 1);

            half3 finRampColor = rampColor + rampColor2;

            // BRDF计算 ------------------------------
                // 法线分布
                float D = D_GGX(NoH, Roughness);
                // 几何遮蔽
                float G = G_Smith(NoV, NoL, Roughness);
                
                // 菲涅尔项
                float3 F = F_Schlick(VoH, F0);


                // 镜面反射项
                float3 specular = (D * G * F) / max(4.0 * NoV * finRampColor, EPSILON);

                // 漫反射项
                float3 kD = max((1.0 - F),0.1) * (1.0 - Metallic);
                float3 diffusePBR = kD * baseColor / PI;

                // 光照组合
                float3 radiance = mainLight.color * mainLight.distanceAttenuation;
                float3 directLight = (diffusePBR + specular) * radiance * finRampColor;

                // 环境光照 ------------------------------
                // 漫反射环境
                float3 ambientDiffuse = SampleSH(N) * baseColor * kD;
                
                // 镜面反射环境
                float3 R = reflect(-V, N);
                float3 prefilteredColor = GlossyEnvironmentReflection(
                    R,
                    Roughness * Roughness,
                    0.5
                );
                float3 ambientSpecular = prefilteredColor * F_SchlickRoughness(NoV, F0, Roughness);
                
                float3 ambientPBR = (ambientDiffuse + ambientSpecular) * 0.5;

                 // 最终颜色
                float3 finalColorPBR = directLight + ambientPBR;

            //finRampColor = saturate(0.5 + 0.5 * finRampColor);

            



            //half3 shadowColor = lerp(baseColor, baseColor * finRampColor, _ShadowIntensity) * _ShaodwColor.rgb;
            //计算漫反射项
            //half3 diffuse = lerp(shadowColor, baseColor, HalfLambertStep);//明部到阴影是在0.423到0.460之间过渡的
            //diffuse = lerp(shadowColor, diffuse, );//将ILM贴图的g通道乘2 用saturate函数将超过1的部分去掉，混合常暗区域（AO）
            //diffuse = lerp(diffuse, baseColor, 1);//将ILM贴图的g通道减0.5乘2 用saturate函数将小于0的部分去掉，混合常亮部分（眼睛）
            //diffuse = diffuse + diffuse;
   
//diffuse = finalColorPBR;
            float shadowAttenuation = mainLight.shadowAttenuation * MainLightRealtimeShadow(input.shadowCoord);
            shadowAttenuation = saturate(0.5 + 0.5 * shadowAttenuation);

             diffuse = (finRampColor * kD * baseColor * mainLight.color + ambientPBR)  * shadowAttenuation * ao / PI;
            }
            else{
                diffuse = 0;
            }
            //diffuse = diffuse +  ambient;
            
            //float3 sdf = SAMPLE_TEXTURE2D(_SDFTex, sampler_SDFTex, input.uv).rgb;
            float sdfThreshold = 0;
            float sdfVlaue = 0;
            float transitionWidth = 0;
            float sdfFace = 0;
            float3 finRampColorface = 0;
            float3 diffuseface = 0;

//       Face


            if (_SDFIDUSE > 0)
            {

            float3 headForward = normalize(_HeadForward - _HeadCenter);
            float3 headRight = normalize(_HeadRight - _HeadCenter);
            float3 headUp = normalize(cross(headForward, headRight));

            float3 fixedLightDir = normalize(lightDirWS - dot(lightDirWS, headUp) * headUp);

            float Sx = dot(fixedLightDir, headRight);
            float Sz = dot(fixedLightDir, -headForward);
            sdfThreshold = atan2(Sx,Sz) / PI;
            sdfThreshold = sdfThreshold > 0 ? (1 - sdfThreshold) : (1 + sdfThreshold);

            float2 sdfUV = input.uv * float2( -1, 1);
            if (dot(fixedLightDir, headRight) > 0)
            {
                sdfUV.x = 1 - sdfUV.x;
            }

            float4 sdfDate = tex2D(_SDFTex, sdfUV);
            float sdfVlaueR = sdfDate.r;
            float sdfVlaueG = sdfDate.g;
            float sdfVlaueB = sdfDate.b;
             sdfVlaue = saturate((sdfVlaueR + sdfVlaueG) * sdfVlaueB);
             sdfVlaue += _FaceshadowOffset;
            transitionWidth = _TransitionWidth;
             sdfFace = smoothstep(
                sdfThreshold - transitionWidth, 
                sdfThreshold + transitionWidth, 
                sdfVlaue
            );


            float shadowMaskface = smoothstep(
                _ShadowThreshold - _ShadowSmoothness,
                _ShadowThreshold + _ShadowSmoothness,
                sdfFace
            );

            float rampUface = sdfFace;
            float rampVface = _RampThreshold;
            float3 rampColorface = SAMPLE_TEXTURE2D(_RampTex, sampler_LinearClamp, float2(rampUface, rampVface)).rgb;

            float rampU2face = sdfFace;
            //float2 rampUV2 = float2(step(HalfLambert , (1 - shadowMask)), 0.5);
            float rampV2face = _RampThreshold;
            float3 rampColor2face = SAMPLE_TEXTURE2D(_RampTex2, sampler_LinearClamp, float2(rampUface, rampVface)).rgb;

            float shadowface = 1 - smoothstep(
    (1 - shadowMaskface) - _ShadowSmoothness, 
    (1 - shadowMaskface) + _ShadowSmoothness, 
    sdfFace
);
            shadowface = smoothstep(0.0, 1.0, shadowface);
            
            rampColor2face = lerp(rampColor2face, float3(1,1,1), 0.7);
            rampColor2face *= shadowface * 0.6;
            
            //rampColor2 = step(rampColor2,shadowMask);


            //half3 shadowColor = lerp(baseColor, baseColor * rampColor2, 1);

             finRampColorface = rampColorface + rampColor2face;

             // BRDF计算 ------------------------------
                // 法线分布
                float D = D_GGX(NoH, Roughness);
                // 几何遮蔽
                float G = G_Smith(NoV, NoL, Roughness);
                
                // 菲涅尔项
                float3 F = F_Schlick(VoH, F0);


                // 镜面反射项
                //float3 specular = (D * G * F) / max(4.0 * NoV * finRampColor, EPSILON);

                // 漫反射项
                float3 kD = max((1.0 - F),0.1) * (1.0 - Metallic);
                float3 diffusePBR = kD * baseColor / PI;

                // 光照组合
               // float3 radiance = mainLight.color * mainLight.distanceAttenuation;
                //float3 directLight = (diffusePBR + specular) * radiance * finRampColor;

                // 环境光照 ------------------------------
                // 漫反射环境
                float3 ambientDiffuse = SampleSH(N) * baseColor * kD;
                
                // 镜面反射环境
                float3 R = reflect(-V, N);
                float3 prefilteredColor = GlossyEnvironmentReflection(
                    R,
                    Roughness * Roughness,
                    0.1
                );
                float3 ambientSpecular = prefilteredColor * F_SchlickRoughness(NoV, F0, Roughness);
                
                float3 ambientPBR = (ambientDiffuse + ambientSpecular) * 0.1;

                 // 最终颜色
                float3 finalColorPBR = directLight + ambientPBR;


             diffuseface = (finRampColorface * kD * baseColor * mainLight.color + ambientPBR) / PI;
             //diffuse = (finRampColor * kD * baseColor * mainLight.color + ambientPBR) * ao / PI;

        }
        else
        {
            sdfThreshold = 0;
            sdfVlaue = 0;
            transitionWidth = 0;
            sdfFace = 0;
             finRampColorface = 0;
             diffuseface = 0;
        }

            //float sdfFace = step(sdfThreshold, sdfVlaue);
             // 过渡区间宽度（可调参数）

            

            

            // float2 sdfUV = float2(sign(dot(fixedLightDir, headRight)),1) * input.uv; 
            // float sdfVlaue = SAMPLE_TEXTURE2D(_SDFTex, sampler_SDFTex, sdfUV).rgb;
            // sdfVlaue += _FaceshadowOffset;
            // float sdfThreshold = 1 - (dot(fixedLightDir, headForward) * 0.5 + 0.5);
            // float sdfFace = step(sdfThreshold, sdfVlaue);

            //float quantizedNdotL = floor(stepNoL * _RampThreshold * 4) / (_RampThreshold * 4);
            //float stepquantizedNdotL = smoothstep(0.0, 0.9 , quantizedNdotL);
            //float2 rampUV = float2(stepquantizedNdotL, 0.5);
            //float3 rampColor = SAMPLE_TEXTURE2D(_RampTex, sampler_LinearClamp, rampUV).rgb;

            //float3 directDiffuse = brdfData.diffuse * mainLight.color * finRampColor / PI;
            //float3 directSpecular = DirectBRDF(brdfData, pixelNormalWS, -lightDirWS, input.viewDirWS) * mainLight.color * saturate(dot(pixelNormalWS, lightDirWS));

            //float3 ambient = UNITY_LIGHTMODEL_AMBIENT.rgb * baseColor;
            //float3 diffuse = directDiffuse + ambient;
            //float3 specular = directSpecular;

            // if (_GGXHair == 1)
            // {
            

            // }



            //float NoL = saturate(dot(pixelNormalWS, lightDirWS));
            //float HalfLambert = saturate(0.5 + 0.5 * NoL);
            //float3 diffuse = baseColor * HalfLambert * lightColor;


            // // 在片元着色器中计算各向异性高光
            // float HdotX = dot(H, input.tangentWS);
            // float HdotY = dot(H, input.bitangentWS);
            // float anisotropy = _Anisotropy;

             float4 AnisoDirectionMap = tex2D(_AnisoDirectionMap, input.uv * _AnisoDirectionMap_ST);
            // float3 noise = AnisoDirectionMap;

            
            // float D1 = AnisoGGXDistribution(NoH, HdotX, HdotY, Roughness, anisotropy);

            // // 结合到BRDF中
            // float3 specularGGX = D1 * F * G / (4.0 * NoL * NoV);

            float aspect = sqrt(1.0 - 0.9 * abs(_Anisotropy));
                float ax = max(0.001, Roughness * Roughness / aspect);
                float ay = max(0.001, Roughness * Roughness * aspect);

                 // 计算D项
                float D1 = D_GGXAniso(ax, ay, NoH, H, T, B);
                
                // 计算G项
                float G1 = V_SmithGGXCorrelatedAniso(ax, ay, ToV, BoV, ToL, BoL, NoV, NoL);

                // 计算F项（Schlick近似）
                float3 F0_GGX = lerp(0.08 * _Specular.xxx, baseColor.rgb, _SpecularTint);
                float3 F_GGX = F0_GGX + (1.0 - F0_GGX) * pow(1.0 - saturate(dot(H, V)), 5.0);

                // 组合BRDF
                float3 specularGGX = (D1 * G1* F_GGX) * 0.25;

                float MetallicGGX = SAMPLE_TEXTURE2D(_ILMTex, sampler_ILMTex, input.uv).x * _MetallicIntensityGGX;

                specularGGX *= 1-MetallicGGX;

                specularGGX *= _specularGGXintensity;
                
                specularGGX *= AnisoDirectionMap;


                specularGGX = specularGGX * _SpecularTint * _SpecularTintIntensity;




            //float3 diffuse = baseColor * NoL * lightColor;
             baseAlpha = SAMPLE_TEXTURE2D(_BaseColorTex, sampler_LinearRepeat, input.uv).a;

            float3 MetallicColor = Metallic * _MetallicColor;

            float3 color = diffuse + diffuseface + EmissionColor + edgeGlow + specularmetllic + SpecularphongColor + specularGGX + (specular * mainLight.color);
            
            //color = saturate(color);
            //color = min(color, 0.7);

            // 厚度贴图采样（反转处理：厚区域值更高）
                half thickness = SAMPLE_TEXTURE2D(_ThicknessMap, sampler_ThicknessMap, input.uv).r;
                thickness = pow(thickness * _DepthContrast, _ThicknessPower);
                // 颜色混合：在厚区域叠加预设颜色
                half3 finalColorsss = lerp(baseColor.rgb, _ThicknessColor.rgb, thickness);

                color = lerp(color, finalColorsss, 0.5);

                color = saturate(color);

                

            return float4 (color, baseAlpha);

            //return float4(DirectLight, baseAlpha);//test
            //return float4(specularGGX, baseAlpha);
        }

        ENDHLSL

        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }
            ZWrite On
            ZTest LEqual
            Cull [_Cull]

            HLSLPROGRAM

            #pragma multi_compile_instancing
            #pragma multi_compile_DOTS_INSTANCING_ON

            #pragma multi_compile_vertex _ _CASTING_PUNCTUAL_LIGHT_SHADOW

            #pragma vertex vert
            #pragma fragment frag

            float3 _LightDirWS;
            float3 _LightPositionWS;
            //float _AlphaClip;

            //#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
           // #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float2 texcoord : TEXCOORD0;
            };

            struct Varyings
            {
                float2 uv : TEXCOORD0;
                float4 positionCS : SV_POSITION;
            };

            float4 GetShadowPositionHClip(Attributes IN)
            {
                float3 positionWS = TransformObjectToWorld(IN.positionOS.xyz);
                float3 normalWS = TransformObjectToWorldNormal(IN.normalOS);

            #if _CASTING_PUNCTUAL_LIGHT_SHADOW
                float3 lightDirWS = normalize(_LightPositionWS - positionWS);
            #else
                float3 lightDirWS = _LightDirWS;
            #endif

                float4 positionCS = TransformWorldToHClip(ApplyShadowBias(positionWS, normalWS, lightDirWS));

            #if UNITY_REVERSED_Z
                positionCS.z = min(positionCS.z, UNITY_NEAR_CLIP_VALUE);
            #else
                positionCS.z = max(positionCS.z, UNITY_NEAR_CLIP_VALUE);
            #endif       
            
                return positionCS;                    
            }

            Varyings vert(Attributes input)
            {
                Varyings output;
                output.uv = input.texcoord;
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                return output;
            }

            float4 frag(Varyings input) : SV_TARGET
            {
                clip(1.0 - _AlphaClip);
                return 0;
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode" = "DepthOnly" }
            ZWrite [_ZWrite]
            ColorMask 0
            Cull [_Cull]

            HLSLPROGRAM

            #pragma multi_compile_instancing
            #pragma multi_compile_DOTS_INSTANCING_ON

            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
            };

            //float _AlphaClip;

            Varyings vert(Attributes input)
            {
                Varyings output;
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                return output;
            }

            float4 frag(Varyings input) : SV_TARGET
            {
                clip(1.0 - _AlphaClip);
                return 0;
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthNormals"
            Tags { "LightMode" = "DepthNormals" }
            ZWrite [_ZWrite]
            Cull [_Cull]

            HLSLPROGRAM

            #pragma multi_compile_instancing
            #pragma multi_compile_DOTS_INSTANCING_ON

            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float4 tangentOS : TANGENT;
                float3 normalOS : NORMAL;
                float2 texcoord : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float4 tangentWS : TEXCOORD2;
            };

            //float _AlphaClip;

            Varyings vert(Attributes input)
            {
                Varyings output = (Varyings)0;

                output.uv = input.texcoord;
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);

                VertexPositionInputs vertexInput = GetVertexPositionInputs(input.positionOS.xyz);
                VertexNormalInputs normalInput = GetVertexNormalInputs(input.normalOS, input.tangentOS);

                float3 viewDirWS = GetWorldSpaceNormalizeViewDir(vertexInput.positionWS);                           
                output.normalWS = half3(normalInput.normalWS);
                float sign = input.tangentOS.w * float(GetOddNegativeScale());
                output.tangentWS = half4(normalInput.tangentWS.xyz,sign);

                return output;               
            }

            half4 frag(Varyings input) : SV_Target
            {
                clip(1.0 - _AlphaClip);
                float3 normalWS = input.normalWS.xyz;
                return half4(NormalizeNormalPerPixel(normalWS),0.0);
            }
            ENDHLSL
            
        }

            Pass
            {
                Name "UniversalForward"
                Tags { "LightMode" = "UniversalForward" }
                ZWrite On
                ZTest LEqual
                Cull [_Cull]
                //  Stencil
                //  {
                //      Ref [_StencilRef]
                //      Comp [_StencilComp]
                //      Pass [_StencilPassOp]
                //      Fail [_StencilFailOp]
                //      ZFail [_StencilZFailOp]
                //  }

            HLSLPROGRAM

            #pragma shader_feature_local _SCREEN_SPACE_RIM
            #pragma shader_feature_local _SCREEN_SPACE_SHADOW
            #pragma shader_feature_local _MATCAP_ON

            #pragma vertex MainVS
            #pragma fragment MainPS

            #pragma multi_compile_fog

            ENDHLSL
            }

            Pass
        {
            Name"UniversalForwardOnly"
            Tags
            {
                "LightMode" = "UniversalForwardOnly"
            }
            Cull Front
            ZWrite On

            HLSLPROGRAM
            #pragma shader_feature_local _OUTLINE_PASS

            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_fog
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
             // 定义 select 函数
             float3 select(int id, float3 e0, float3 e1, float3 e2, float3 e3, float3 e4)
             {
                 return id == 0 ? e0 : (id == 1 ? e1 : (id == 2 ? e2 : (id == 3 ? e3 : e4)));
             }
            // If your project has a faster way to get camera fov in shader, you can replace this slow function to your method.
            // For example, you write cmd.SetGlobalFloat("_CurrentCameraFOV",cameraFOV) using a new RendererFeature in C#.
            // For this tutorial shader, we will keep things simple and use this slower but convenient method to get camera fov
            float GetCameraFOV()
            {
                //https://answers.unity.com/questions/770838/how-can-i-extract-the-fov-information-from-the-pro.html
                float t = unity_CameraProjection._m11;
                float Rad2Deg = 180 / 3.1415;
                float fov = atan(1.0f / t) * 2.0 * Rad2Deg;
                return fov;
            }
            float ApplyOutlineDistanceFadeOut(float inputMulFix)
            {
                //make outline "fadeout" if character is too small in camera's view
                return saturate(inputMulFix);
            }
            float GetOutlineCameraFovAndDistanceFixMultiplier(float positionVS_Z)
            {
                float cameraMulFix;
                if(unity_OrthoParams.w == 0)
                {
                    ////////////////////////////////
                    // Perspective camera case
                    ////////////////////////////////

                    // keep outline similar width on screen accoss all camera distance       
                    cameraMulFix = abs(positionVS_Z);

                    // can replace to a tonemap function if a smooth stop is needed
                    cameraMulFix = ApplyOutlineDistanceFadeOut(cameraMulFix);

                    // keep outline similar width on screen accoss all camera fov
                    cameraMulFix *= GetCameraFOV();       
                }
                else
                {
                    ////////////////////////////////
                    // Orthographic camera case
                    ////////////////////////////////
                    float orthoSize = abs(unity_OrthoParams.y);
                    orthoSize = ApplyOutlineDistanceFadeOut(orthoSize);
                    cameraMulFix = orthoSize * 50; // 50 is a magic number to match perspective camera's outline width
                }

                return cameraMulFix * 0.00005; // mul a const to make return result = default normal expand amount WS
            }
            // Push an imaginary vertex towards camera in view space (linear, view space unit), 
            // then only overwrite original positionCS.z using imaginary vertex's result positionCS.z value
            // Will only affect ZTest ZWrite's depth value of vertex shader

            // Useful for:
            // -Hide ugly outline on face/eye
            // -Make eyebrow render on top of hair
            // -Solve ZFighting issue without moving geometry
            float4 NiloGetNewClipPosWithZOffset(float4 originalPositionCS, float viewSpaceZOffsetAmount)
            {
                if(unity_OrthoParams.w == 0)
                {
                    ////////////////////////////////
                    //Perspective camera case
                    ////////////////////////////////
                    float2 ProjM_ZRow_ZW = UNITY_MATRIX_P[2].zw;
                    float modifiedPositionVS_Z = -originalPositionCS.w + -viewSpaceZOffsetAmount; // push imaginary vertex
                    float modifiedPositionCS_Z = modifiedPositionVS_Z * ProjM_ZRow_ZW[0] + ProjM_ZRow_ZW[1];
                    originalPositionCS.z = modifiedPositionCS_Z * originalPositionCS.w / (-modifiedPositionVS_Z); // overwrite positionCS.z
                    return originalPositionCS;    
                }
                else
                {
                    ////////////////////////////////
                    //Orthographic camera case
                    ////////////////////////////////
                    originalPositionCS.z += -viewSpaceZOffsetAmount / _ProjectionParams.z; // push imaginary vertex and overwrite positionCS.z
                    return originalPositionCS;
                }
            }

            struct Attributes
            {
                float4 positionOS   : POSITION;
                float4 tangentOS    : TANGENT;
                float3 normalOS     : NORMAL;
                float2 texcoord     :TEXCOORD0;
                float2 texcoord1    :TEXCOORD1;
                float2 uv: TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS  : SV_POSITION;
                float fogFactor    : TEXCOORD1;
                float2 uv: TEXCOORD0;
            };

            //sampler2D _BaseColorTex;;
            float _OutLineWidth;
            float _MaxOutlineZoffset;
            float4 _OutlineColor;
            //float _MaterialIDUSE; // 添加材质ID变量
            //float4 _OutlineColor2;
            //float4 _OutlineColor3;
            //float4 _OutlineColor4;
            //float4 _OutlineColor5;
            //TEXTURE2D(_OtherDataTex);
            //SAMPLER(sampler_OtherDataTex);
            
            //CBUFFER_START(UnityPerMaterial)
            //float4 _Color;
            //float4 _BaseMap_ST;
            //CBUFFER_END

            Varyings vert(Attributes input)
            {   
                //#if !_OUTLINE_PASS
                //return (Varyings)0;
                //#endif

                VertexPositionInputs positionInputs = GetVertexPositionInputs(input.positionOS.xyz);
                VertexNormalInputs normalInputs = GetVertexNormalInputs(input.normalOS,input.tangentOS);

                float width = _OutLineWidth;
                width *= GetOutlineCameraFovAndDistanceFixMultiplier(positionInputs.positionVS.z);

                float3 positionWS = positionInputs.positionWS.xyz;
                positionWS += normalInputs.normalWS * width;

                Varyings output = (Varyings)0;
                output.positionCS = NiloGetNewClipPosWithZOffset(TransformWorldToHClip(positionWS),_MaxOutlineZoffset);
                output.uv = input.texcoord;
                output.fogFactor = ComputeFogFactor(positionInputs.positionCS.z);

                //Varyings OUT;

                //OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                //OUT.uv = TRANSFORM_TEX(IN.uv, _BaseMap);

                return output;
            }

            float4 frag(Varyings input) : SV_Target
            {
                //float4 texel = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.uv);
                float3 outlineColor = 0;

                outlineColor = _OutlineColor.rgb;

                // 使用 _OtherDataTex 的 x 通道选择描边颜色
                // float4 otherData = SAMPLE_TEXTURE2D(_OtherDataTex, sampler_OtherDataTex, input.uv);
                // int materialid = max(0,4 - floor(otherData.x *5)); // 将 x 通道值映射到 0-5的整数范围
                // outlineColor = select(materialid, _OutlineColor.rgb, _OutlineColor2.rgb, _OutlineColor3.rgb, _OutlineColor4.rgb, _OutlineColor5.rgb);
                

                //float3 baseMapColor = SAMPLE_TEXTURE2D(_BaseColorTex, sampler_LinearRepeat, input.uv);
                outlineColor *= 0.1;

                float4 color = float4(outlineColor,1);
                color.rgb = MixFog(color.rgb, input.fogFactor);

                return color;
                //return float4(baseMapColor,1);
            }
            ENDHLSL
        }

            
        }
        Fallback "Diffuse"
    }

        
