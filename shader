Shader "Unlit/pbr"
{
    Properties
    {
        _BaseColorTex("Base Color Tex", 2D) = "white"{}
        _Color("Color", Color) = (1,1,1,1)
        _NormalTex("Normal Tex", 2D) = "bump" {}
        _BumpScale("Bump Scale", Range(-5,5)) = 1
        
        _SDFTex("SDF Tex", 2D) = "white" {}
        _MaterialIDUSE("Material ID USE", Range(0,4)) = 0
        _SDFIDUSE("SDF ID USE", Range(0,4)) = 0
        _RampTex("Ramp Tex", 2D) = "white" {}
        _RampTex2("Ramp Tex2", 2D) = "white" {}
        _RoughnessIntensity("Roughness Intensity", Range(0,5)) = 0
        _SmoothnessIntensity("Smoothness Intensity", Range(0,1)) = 0
        _SpecularTintIntensity("Specular Tint Intensity", Range(0,1)) = 0
        _SHIntensity("SH Intensity", Range(0,1)) = 0
        _aoUsage("AO Usage", Range(0,1)) = 0
        _AmbientIntensity("Ambient Intensity", Range(0,1)) = 0
        _Ambientpower("Ambient Power", Range(0,1)) = 0
        _FresnelPower("Fresnel Power", Range(0, 10)) = 5.0   // 控制边缘宽度
        _FresnelScale("Fresnel Scale", Range(0, 1)) = 0.5    // 控制强度
        _EdgeColor("Edge Color", Color) = (1, 0.5, 0, 1)     // 边缘光颜色

         _ThicknessMap("Thickness Map", 2D) = "white" {}
        _ThicknessColor("Thickness Color", Color) = (0.8,0.5,0.4,1) // 模拟皮下血管颜色
        _ThicknessPower("Thickness Power", Range(0,5)) = 2.0
        _DepthContrast("Depth Contrast", Range(0,5)) = 1.5
        _NoseColor("Nose Color", Color) = (1,0.5,0.4,1) // 鼻子高光颜色
        _NoseIntensity("Nose Intensity", Range(0,10)) = 1
        _MouthTex("Mouth Mask", 2D) = "white" {}
        _Gloss("Gloss", Range(0,1)) = 0.5

        _LutTex("Lut Tex", 2D) = "white" {}



        [HideInInspector]_HeadCenter("Head Center", Vector) = (0,0,0)
        [HideInInspector]_HeadForward("Head Forward", Vector) = (0,0,0)
        [HideInInspector]_HeadRight("Head Right", Vector) = (0,0,0)

        _FaceshadowOffset("Face shadow Offset", Range(-1,1)) = 0.1
        _TransitionWidth("Transition Width", Range(0,1)) = 0.05
        //_EmissionIntensity("Emission Intensity", Range(0,1)) = 0

        _SpecularTint("Specular Tint", Color) = (1,1,1,1)
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
#define LUT_WIDTH  1024
#define LUT_HEIGHT 32
#define LUT_DEPTH  32   // B 通道分层数
#define R_SEGMENTS 32   // R 分段数
#define G_SEGMENTS 32   // 每个 R 分段内的 G 值数
        #define kDielectricSpec half4(0.04, 0.04, 0.04, 1.0 - 0.04) // standard dielectric reflectivity coef at incident angle (= 4%)
        


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
            sampler2D _RampTex;
            sampler2D _RampTex2;
            sampler2D _MouthTex;

            TEXTURE2D (_LutTex);
            SAMPLER (linear_clamp_sampler);
            //TEXTURE2D(_NormalTex);
            //TEXTURE2D(_OcclusionTex);
            //float4 _MetallicColor;
            //float4 _Color;
            // float _ShadowThreshold;
            // float _ShadowSmoothness;
            //float _ShadowIntensity;
            float _SHIntensity;
            float _aoUsage;
            float _AmbientIntensity;
            float _Ambientpower;
            float3 _HeadCenter;
            float3 _HeadForward;
            float3 _HeadRight;
            float _FaceshadowOffset;
            float _TransitionWidth;
            float3 _EdgeColor;
            float _FresnelScale;
            float4 _NoseColor;
            float _NoseIntensity;
            float _Gloss;
            

            
            //SAMPLER(sampler_LinearRepeat);
            
            //sampler2D _OcclusionTex;
            float _BumpScale;
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
            float _AlphaClip;
            float _MaterialIDUSE;
            float _SDFIDUSE;
            float _FresnelPower;

        CBUFFER_END

        // --- LUT 采样函数 ---
float3 ApplySkinLUT(float3 originalColor)
{
    // 将颜色限制在 [0, 1] 范围
    float3 color = saturate(originalColor);
    
    // --- 计算 B 通道索引（决定行）---
    float g = color.g * (LUT_DEPTH - 1);
    int bIndex = (int)floor(g);
    bIndex = clamp(bIndex, 0, LUT_DEPTH - 1); // 确保不越界
    
    // --- 计算 R 通道索引（决定横向分段）---
    float b = color.b * (R_SEGMENTS - 1);
    int rIndex = (int)floor(b);
    rIndex = clamp(rIndex, 0, R_SEGMENTS - 1);
    
    // --- 计算 G 通道索引（决定分段内位置）---
    float r = color.r * (G_SEGMENTS - 1);
    int gIndex = (int)floor(r);
    gIndex = clamp(gIndex, 0, G_SEGMENTS - 1);
    
    // --- 计算 UV 坐标 ---
    // U轴：R分段起始位置 + G偏移 → 转换为 [0,1]
    float u = (rIndex * G_SEGMENTS + gIndex + 0.5) / LUT_WIDTH;
    // V轴：B行位置 → 转换为 [0,1]
    float v = (bIndex + 0.5) / LUT_HEIGHT;
    
    return SAMPLE_TEXTURE2D(_LutTex,linear_clamp_sampler, float2(u,v)).rgb;
}


        // UE5改进的各向异性分布函数
            float D_GGXAnisoUE5(float at, float ab, float ToH, float BoH, float NoH)
            {
                float ToH2 = ToH * ToH;
                float BoH2 = BoH * BoH;
                float NoH2 = NoH * NoH;
                float denominator = ToH2/(at*at) + BoH2/(ab*ab) + NoH2;
                denominator = PI * at * ab * (denominator * denominator);
                return 1.0 / denominator;
            }

            // UE5改进的几何项
            float V_SmithGGXCorrelatedAnisoUE5(float at, float ab, float ToV, float BoV, float ToL, float BoL, float NoV, float NoL)
            {
                float lambdaV = NoL * sqrt(NoV * (NoV - NoV * at) + at);
                float lambdaL = NoV * sqrt(NoL * (NoL - NoL * ab) + ab);
                return 0.5 / (lambdaV + lambdaL);
            }

           // 修改旋转函数
float2 RotateAnisotropy(float2 dir, float rotation)
{
    float angle = rotation * 2.0 * PI;
    float sinRot, cosRot;
    sincos(angle, sinRot, cosRot);
    return float2(
        dir.x * cosRot - dir.y * sinRot,
        dir.x * sinRot + dir.y * cosRot
    );
}

float3 CalculateSSR(float3 reflection, float3 positionWS, float roughness)
{
    #if defined(_SSR_ENABLED)
    // 修正1：使用正确的屏幕UV计算
    float2 screenUV = (input.screenPos.xy / input.screenPos.w);
    #if UNITY_UV_STARTS_AT_TOP
    screenUV.y = 1.0 - screenUV.y;
    #endif
    
    // 修正2：使用URP内置深度采样方法
    float rawDepth = SampleSceneDepth(screenUV);
    float3 rayStart = GetCameraPositionWS();
    float3 rayDir = normalize(reflection);
    
    // 修正3：调整步进参数
    const int steps = 32; // 增加步数
    const float stepSize = 0.2; // 增大步长
    const float thickness = 0.1; // 增加厚度容差
    
    for(int i=1; i<=steps; i++)
    {
        float3 rayEnd = positionWS + rayDir * i * stepSize;
        float4 projEnd = TransformWorldToHClip(rayEnd);
        float2 uvEnd = projEnd.xy / projEnd.w * 0.5 + 0.5;
        
        // 处理平台差异
        #if UNITY_UV_STARTS_AT_TOP
        uvEnd.y = 1.0 - uvEnd.y;
        #endif
        
        // 跳过屏幕外坐标
        if(uvEnd.x < 0 || uvEnd.x > 1 || uvEnd.y < 0 || uvEnd.y > 1) 
            continue;
        
        float sceneDepth = SampleSceneDepth(uvEnd);
        float3 scenePos = ComputeWorldSpacePosition(uvEnd, sceneDepth, UNITY_MATRIX_I_VP);
        
        // 使用距离+法线双重检测
        float surfaceDiff = distance(rayEnd, scenePos);
        float3 sceneNormal = SampleSceneNormals(uvEnd);
        float normalCheck = saturate(dot(sceneNormal, -rayDir));
        
        if(surfaceDiff < stepSize * thickness && normalCheck > 0.3)
        {
            // 使用Mipmap优化采样
            return SAMPLE_TEXTURE2D_LOD(_CameraOpaqueTexture, sampler_CameraOpaqueTexture, uvEnd, roughness * 8).rgb;
        }
    }
    #endif
    return 0;
}

        


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
            float4 positionWSAndFogFactor  : TEXCOORD1; // Stores world-space position (xyz) and fog factor (w)
            float3 normalWS                : TEXCOORD2;
            float4 tangentWS               : TEXCOORD3;
            float3 bitangentWS             : TEXCOORD4;
            float3 viewDirWS               : TEXCOORD5;
            float4 positionCS              : SV_POSITION;
            float3 SH                      : TEXCOORD6;  
            float3 positionWS              : TEXCOORD7;       
            float4 shadowCoord             : TEXCOORD8; // New shadow coordinates
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

            float3 lightColor = mainLight.color;
            float3 diffuse = 0;


             
         
            float3 ambient = input.SH.rgb * _Ambientpower;
            //ambient *= lerp(1, ao, _aoUsage);
            ambient = lerp(ambient, baseColor, _AmbientIntensity);
            
            

//          body
            if (_SDFIDUSE == 0)
            {
            float  NoL = saturate(dot(pixelNormalWS, lightDirWS));
            NoL = smoothstep(0.0, 1.0, NoL);
            NoL = saturate(0.5 + 0.5 * NoL);
            //float HalfLambert = saturate(0.5 + 0.5 * stepNoL);

            // 菲涅尔计算
                //float3 normalWS = normalize(input.normalWS);
                float3 viewDirWS = normalize(input.viewDirWS);
                float fresnel = pow(1.0 - saturate(dot(pixelNormalWS, normalize(input.viewDirWS))), _FresnelPower);
                fresnel = smoothstep(0.3,0.8,fresnel);
                float3 edgeGlow = lightColor * _EdgeColor.rgb * fresnel * _FresnelScale ;
                edgeGlow *= lerp(baseColor,float3(1,1,1),0.9);
                edgeGlow *= NoL;


            //float thickness = tex2D(_RampTex, input.uv).a; // 皮肤厚度图
            float2 sssUV = float2(NoL * 0.3 + 0.5, 0.5); 
            float3 sssColor = tex2D(_RampTex, sssUV).rgb; // 透光颜色（如红色）
            sssColor = saturate(sssColor);
            //finalColor += sssColor ;
            float3 skinColor = ApplySkinLUT(baseColor);

            baseColor = lerp(baseColor, skinColor, 0.5);
                
                
                
                float3 diffuse1 = baseColor / PI;

                // 合并光照
                float3 radiance = _MainLightColor.rgb * NoL;
                float3 color = diffuse1 * sssColor * radiance;
                float3 Lutcolor = ApplySkinLUT(color);
                color = lerp(color, Lutcolor, 0.5);

                

                // 添加环境光照
                float3 ambient = SampleSH(N) * baseColor.rgb;
                


            
             
        
            float shadowAttenuation = mainLight.shadowAttenuation * MainLightRealtimeShadow(input.shadowCoord);
            shadowAttenuation = saturate(0.5 + 0.5 * shadowAttenuation);

            diffuse =  color + ambient * shadowAttenuation + edgeGlow;
            //diffuse = color;
            }
            else{
                diffuse = 0;
            }
            
            float sdfThreshold = 0;
            float sdfVlaue = 0;
            float transitionWidth = 0;
            float sdfFace = 0;
            float3 finRampColorface = 0;
            float3 diffuseface = 0;
            float FaceSSSTex = 0;

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
             // GLSL/HLSL示例
            float originalValue = sdfVlaueR;
            float epsilon = 0.1; // 容差值
            float modifiedValue = (abs(originalValue) < epsilon) ? 0.1 : originalValue;
            //float modifiedValue = (originalValue == 0.0) ? 0.009 : originalValue;
            sdfVlaue = modifiedValue;
            
             sdfVlaue += _FaceshadowOffset;
            transitionWidth = _TransitionWidth;

             sdfFace = smoothstep(
                sdfThreshold - transitionWidth, 
                sdfThreshold + transitionWidth, 
                sdfVlaue);

            sdfFace += _FaceshadowOffset;
            sdfFace = saturate(sdfFace);

            //FaceSSSTex = SAMPLE_TEXTURE2D(_ThicknessMap, sampler_ThicknessMap, input.uv).r;

            

            float2 faceRampUV = float2(sdfFace * 0.3 + 0.5,0.5);

            float3 faceRampColor = tex2D(_RampTex2,faceRampUV).rgb;
            faceRampColor = saturate(faceRampColor);

            float3 lutface = ApplySkinLUT(baseColor);
            baseColor = lerp(baseColor, lutface, 0.5);
            
            // 采样脸部遮罩贴图
            float maskValue = SAMPLE_TEXTURE2D(_ThicknessMap, sampler_ThicknessMap, input.uv).r; // 使用R通道作为遮罩值

            float3 faceSSS = lerp(float3(1,1,1),_ThicknessColor.rgb,maskValue);

            //鼻子高光
            float noseHighlight = SAMPLE_TEXTURE2D(_ThicknessMap, sampler_ThicknessMap, input.uv).a;
            noseHighlight *= sdfVlaueB;
            // 分离左右高光（UV.x < 0.5为左，>0.5为右）
               // 优化后代码（严格分割）
                float leftMask = (input.uv.x > 0.5) ? noseHighlight : 0.0; // 仅右侧区域有效
                float rightMask = (input.uv.x <= 0.5) ? noseHighlight : 0.0; // 仅左侧区域有效

                // 计算视角与光源的夹角方向（正值为右，负值为左）
                float3 viewLightDir = H;
                float side = dot(normalize(sdfFace), viewLightDir);

                 // 动态选择高光侧
                float finalMask = (side > 0.0) ? leftMask : rightMask; // 完全二选一

                float3 NoseHighlightColor = _NoseIntensity * _NoseColor * finalMask; // 仅在选定侧显示高光

                 // 菲涅尔计算
                //float3 normalWS = normalize(input.normalWS);
                float3 viewDirWS = normalize(input.viewDirWS);
                float fresnel = pow(1.0 - saturate(dot(pixelNormalWS, normalize(input.viewDirWS))), _FresnelPower);
                fresnel = smoothstep(0.3,0.8,fresnel);
                float3 edgeGlow = lightColor * _EdgeColor.rgb * fresnel * _FresnelScale * finalMask;
                edgeGlow *= lerp(baseColor,float3(1,1,1),0.9);
                edgeGlow *= NoL;


            float _MaxOffset = 0.03; // 控制左右最大偏移范围
            float Vx = V.x;
            float clampedOffset = clamp(Vx, -_MaxOffset, _MaxOffset); // 双向钳制
            float2 mouthUV = float2(input.uv.x + clampedOffset, input.uv.y); // 调整UV坐标范围
            float mouth = tex2D(_MouthTex, mouthUV).r; // 采样嘴巴贴图
            // 在切线空间中偏移UV（模拟高光移动）

                // 转换到切线空间
            float3 lightDirTS = mul(TBN, lightDirWS);
            float3 viewDirTS = mul(TBN, viewDirWS);
            
            // 计算半角向量（Blinn-Phong优化）
            float3 halfDirTS = normalize(lightDirTS + viewDirTS);
            
            // 采样高光Mask（唇部区域）
            //float specMask = tex2D(_SpecMask, input.uv).r;
             float2 uvOffset = viewDirTS.xy * 0.1; // 调整0.1控制偏移量
            float specMask = tex2D(_MouthTex, input.uv + uvOffset).r;
            
            // 计算高光强度（基于切线空间半角向量）
            float specPower = pow(_Gloss, 2.0); // 转换为非线性范围
            float NdotH = dot(sdfFace, halfDirTS);
            float spec = pow(saturate(NdotH), specPower) * 2;

            float finalSpec = spec * mouth; // 仅在嘴巴区域显示高光
                

                
                
                
                
                float3 diffuseface1 = baseColor / PI;

                // 合并光照
                float3 radianceface = _MainLightColor.rgb * sdfFace;
                float3 colorface = diffuseface1 * faceRampColor * radianceface * faceSSS;
                float3 Lutcolorface = ApplySkinLUT(colorface);
                colorface = lerp(colorface, Lutcolorface, 0.5);
                
                

                

                // 添加环境光照
                float3 ambient = SampleSH(sdfFace) * baseColor.rgb;


            
             
        
            // float shadowAttenuation = mainLight.shadowAttenuation * MainLightRealtimeShadow(input.shadowCoord);
            // shadowAttenuation = saturate(0.5 + 0.5 * shadowAttenuation);

            diffuseface = colorface +  ambient + NoseHighlightColor + finalSpec;

            //diffuseface = finalSpec.xxx;

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


            // 厚度贴图采样（反转处理：厚区域值更高）
                half thickness = SAMPLE_TEXTURE2D(_ThicknessMap, sampler_ThicknessMap, input.uv).r;
                thickness = pow(thickness * _DepthContrast, _ThicknessPower);
                // 颜色混合：在厚区域叠加预设颜色
                //half3 finalColorsss = lerp(baseColor.rgb, _ThicknessColor.rgb, thickness);

                //color = lerp(color, finalColorsss, 0.5);
                float3 color = diffuse + diffuseface;

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

        
