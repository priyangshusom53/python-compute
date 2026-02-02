#ifndef GLTF_LOADER_H
#define GLTF_LOADER_H

#include "vector.h"
#include "point.h"
#include "normal.h"
#include "transformation.h"
#include "mesh.h"
#include "material.h"

#include<stdexcept>
#include<iostream>
#include<vector>
#include<string>
#include<queue>
#include<map>
#include<fstream>
#include<memory>

#include "tiny_gltf.h"

#pragma region GEOMETRY AND VERTEX DATA
/*
*	Accessors:
*		contains index and offset to Bufferview and type of data(Ex:VEC3, SCALAR)
*		Example:
*			{
*            "bufferView": 0,
*            "byteOffset": 0,
*            "componentType": 5123,
*            "count": 12636,
*            "max": [
*                4212
*            ],
*            "min": [
*                0
*            ],
*            "type": "SCALAR"
*            }
* 
*       "byteOffset" property specifies the location of the
*       first data element within the referenced buffer view.
* 
*		first element = bufferViews[bufferView][byteOffset]
*		num of elements = count
*		componentType: { float, signed byte, unsigned byte, unsigned int }
*		type(data type): {SCALAR, VEC2, VEC3, VEC4, MAT4}
*		Example:
*			"accessors": [
*				{
*					"bufferView": 1,
*					"byteOffset": 7032,
*					"componentType": 5126,
*					"count": 585,
*					"type": "VEC3"
*				}
*			]
*
*	Bufferviews:
*		contains index to buffer and "byteOffset" for
*       starting index and "byteLength" in bytes
*		optionally contains "byteStride" for vertex attribute data
*		Example:
*			"bufferViews": [
*				{
*					"buffer": 0,
*					"byteLength": 17136,
*					"byteOffset": 620
*				}
*			]
*
*/

/*
*	Meshes:
*		A mesh is defined as array of "primitives"
*		each "primitive" is like submesh each contain own data and
*		material info
*		Example:
*			"meshes": [
*               {
*                   "primitives": [
                        {
                            "attributes": {
                                "NORMAL": 23,
                                "POSITION": 22,
                                "TANGENT": 24,
                                "TEXCOORD_0": 25
                            },
                            "indices": 21,
                            "material": 3,
                            "mode": 4
                        }
                    ]
                }
            ]
*       each "primitive" contain "attributes" which contain vertex
*       attributes(Ex: POSITION, NORMAL) which have index to
*       "accessors", "indices" is index to "accessors" for index data
*       
*       "attributes" types:
*           POSITION: VEC3, float
*           NORMAL: VEC3, float
*           TANGENT: VEC4, float
*           TEXCOORD_n: VEC2,(float/unsigned byte normalized/unsigned short normalized
*
*       if "indices" is not present "count" of attribute "accessor"
*       gives number of vertices to render
*       
*       "material": index to "materials" array
*       "mode":
*           corresponds to topology types
*       types are:
*           Points:
*               each vertex
*           Line Strips:
*               defined by each vertex and following vertex,{vi, vi+1}
*           Line Loops:
*               same as strips last vertex is first vertex is extra line
*           Lines:
*               every two vertices pi = {v2i, v2i+1}
*           Triangles:
*               every 3 set of vertices pi = {v3i, v3i+1, v3i+2}
*           Triangle Strips:
*               One triangle primitive is defined by each vertex and the two vertices that follow it
*               pi = {vi, vi+(1+i%2), vi+(2-i%2)}
*
*   Nodes:
*       nodes can contain 0 or 1 mesh
*       nodes contain own "transform" either TRS or MAT4
*
*       more than 1 node can contain same "mesh"
*
*       "mesh" is index to "meshes" array
*       Example:
*           "nodes": [
                {
                    "mesh": 11
                },
                {
                    "mesh": 11,
                    "translation": [
                        -20,
                        -1,
                         0
                    ]
                }
            ]
        here two nodes contain same mesh
*/
#pragma endregion

#pragma region TEXTURE
/*
*   Texture:
*       Example:
            {
                "textures": [
                    {
                        "sampler": 0,
                        "source": 2
                    }
                ]
            }
            texture contain "source" index to "images" array
            "sampler" index to "samplers" array
*   Image:
*       contains URI to image file(png or jgeg) or 
*       index to "bufferView" with "mimeType": "image/(jpeg or png)"
* 
*   Sampler:
*       Samplers are stored in the samplers array of the asset. 
*       Each sampler specifies filtering and wrapping modes.
*       
*       Filtering:
*           Filtering modes control texture’s magnification and minification.
*           Magnification modes:
*               Nearest: gives color of nearest pixel coordinate
*               Linear: linear computes weighted sum of adjacent pixels
*           
*           Minification modes:
*               Nearest: gives color of nearest pixel coordinate
*               
*               Linear: linear computes weighted sum of adjacent pixels
* 
*               Nearest-mipmap-nearest: selects nearest mipmap then nearest pixel from that mipmap
*               
*               Linear-mipmap-nearest: weighted sum of pixels from nearest mipmap
*       
*               Nearest-mipmap-linear: nearest pixel from weighted sum of two mipmaps
*           
*               Linear-mipmap-linear: weighted sum of adjacent pixels from weighted sum of two mipmaps
*               
*       Wrapping:
*           Sampler’s wrapping modes define how to handle texture 
*           coordinates that are negative or greater than or 
*           equal to 1.0, independently for both directions X(S) and Y(T),
*           
*           modes:
*               Repeat: Only the fractional part of texture coordinates is used
*               Example:2.2 maps to 0.2; -0.4 maps to 0.6.
*           
*               Mirrored Repeat: works as repeat but flips the direction when the 
*               integer part (truncated towards -infinity) is odd.
*               Example:2.2 maps to 0.2; -0.4 is treated as 0.4
* 
*               Clamp to edge:Texture coordinates with values outside the image are 
*               clamped to the closest existing image texel at the edge.              
*/
#pragma endregion

#pragma region MATERIAL
/*
*   Materials:
*       contains PBR materials in "materials" array
*       Example:
*           "materials": [
                {
                    "name": "gold",
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [ 1.000, 0.766, 0.336, 1.0 ],
                        "metallicFactor": 1.0,
                        "roughnessFactor": 0.0
                    }
                }
            ]
*       
*       "pbrMetallicRoughness" properties:
*           "baseColorFactor": diffuse color for non metals and F0 for metals
*               also acts as factor for "baseColorTexture"
*           "baseColorTexture":
*               Example:
*                   "baseColorTexture": {
                        "index": 0,
                        "texCoord": 1
                    }
*               "index" is index to "textures" array
*               "texCoord": optional texCoord channel 0 if not mentioned
*               format: 8 bit sRGB(RGB or RGBA)
*           "metallicRoughnessTexture":
*               green channel: roughness
*               blue channel:  metalness
*           
*       "normalTexture": seperate texture outside "pbrMetallicRoughness"
*           for normals
*           format: RGB
*       "occlusionTexture": [0.0-1.0] occlusion value
*       
*       "emissiveFactor": factor for "emissive"
*       "emissive": vlaue of light 
*           format: 8 bit sRGB(RGB) texture
*/
#pragma endregion

struct Node {
    Transform localToWorld;
    std::shared_ptr<Node> parent;   // null for root nodes
    std::vector<std::shared_ptr<Node>> children;// empty for leaf nodes 
    std::shared_ptr<TriangleMesh> mesh; // null if no mesh
    std::string name;
};

struct Scene {

    std::vector<int> roots;
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<std::shared_ptr<TriangleMesh>> triangleMeshes;
    std::vector<std::shared_ptr<PBRMaterial>> materials;
    std::string name;
};


/*
*   Used in CPU only
*/
class GLTFLoader {

    void Load(const std::string& path) {

        if (!std::ifstream(path.c_str()).good()) {
            std::cout << '\n' << "File not exist";
            return;
        }

        tinygltf::Model model;
        tinygltf::TinyGLTF gltf_ctx;
        std::string _path = path;
        std::string error;
        std::string warn;

        // My Scene types
        std::shared_ptr<Scene> scene;   // Can be more than 1 scene

        bool ret = false;

        size_t dotIdx = 0;
        dotIdx = _path.find_last_of('.');
        if (dotIdx == std::string::npos) {
            // replace with logger or printf
            std::cout <<'\n'<< "Not a .gltf or .glb file";
            return;
        }
        else if(_path.substr(dotIdx+1)=="gltf") {
            std::cout << '\n' << "Opening " 
                << _path.substr((_path.find_last_of('\\')+1))<<" as .gltf";

            // Load .gltf file
            ret = gltf_ctx.LoadASCIIFromFile(&model, &error, &warn, path);

        }
        else if (_path.substr(dotIdx + 1) == "glb") {
            std::cout << '\n' << "Opening "
                << _path.substr((_path.find_last_of('\\') + 1)) << " as .glb";

            ret = gltf_ctx.LoadBinaryFromFile(&model, &error, &warn, path);
        }
        else {
            std::cout << '\n' << "File is not a .gltf or .glb file";
            return;
        }

        if (ret == false) {
            std::cout << '\n'<<"Couldn't open file"
                <<'\n' << "Error: " << error
                << '\n' << "Warning: " << warn;
            return;
        }

        auto tinygltfAccessors = model.accessors;
        

        // Store meshes
        for (int i = 0; i < model.meshes.size(); ++i) {

        }

        scene = std::make_shared<Scene>();
        auto tinygltfScene = model.scenes[0];
        scene->name = tinygltfScene.name;
        scene->roots = tinygltfScene.nodes;

        auto tinygltfNodes = model.nodes;
        std::queue<int> nodesQue;
        for (int i = 0; i < tinygltfScene.nodes.size(); ++i) {
            
            nodesQue.push(tinygltfScene.nodes[i]);
            while (!nodesQue.empty()) {
                int nodeidx = nodesQue.front();
                nodesQue.pop();
                auto nodeToProcess = tinygltfNodes[nodeidx];
                if (nodeToProcess.children.size() > 0) {
                    for (int ch = 0; ch < nodeToProcess.children.size(); ++ch) {
                        nodesQue.push(nodeToProcess.children[ch]);
                    }
                }
                else {// Create Node

                }
            }
        }

    }

    void LoadMeshes(
        const std::vector<tinygltf::Buffer>& bufs,
        const std::vector<tinygltf::BufferView>& bufViews,
        const std::vector<tinygltf::Accessor>& accessors,
        tinygltf::Mesh& tglMesh,
        std::vector<std::shared_ptr<TriangleMesh>>& meshes
        ) {

        auto& primitives = tglMesh.primitives;
        // Each primitives are like sub meshes
        for (int i = 0; i < primitives.size(); ++i) {
            
            auto& attributes = primitives[i].attributes;
            
            int pIdx = attributes["POSITION"];
            auto& pAcsr = accessors[pIdx];
            auto& pBufV = bufViews[pAcsr.bufferView];
            
            int nIdx = 
                (attributes.find("NORMAL") == attributes.end())?
                -1:attributes["NORMAL"];
            bool hasNormal = nIdx >= 0 ? true : false;

            int tex_0_Idx =
                (attributes.find("TEXCOORD_0") == attributes.end()) ?
                -1 : attributes["TEXCOORD_0"];
            bool hasTexCoord_0 = tex_0_Idx >= 0?true : false;

            int indicesIndex = primitives[i].indices;
            bool hasIndices = indicesIndex >= 0 ? true : false;
            int indicesCount = 0;
            // attribute accessor.count/3 is indicesCount
            if (indicesIndex < 0) indicesCount = pAcsr.count/3;

            int materialIndex = primitives[i].material;

            std::vector<Point3f> positions(pAcsr.count);
            std::vector<Vector3i> indices;
            std::vector<Normal3f> normals;
            std::vector<Vector2f> texCoords_0;
            
            if (hasIndices) indices.resize(accessors[indicesIndex].count);
            if (hasTexCoord_0) texCoords_0.resize(accessors[tex_0_Idx].count);

            auto& pBuf = bufs[pBufV.buffer];
            int byteOffset = pBufV.byteOffset +
                pAcsr.byteOffset;
            const float* positionData = reinterpret_cast<const float*>(&pBuf.data[byteOffset]);
            for (int v = 0; v < positions.size(); ++v) {
                
                Point3f p(positionData[3 * v], positionData[3 * v + 1], positionData[3 * v + 2]);
                positions[v] = p;
            }

            if (hasNormal) {
                auto& nAcsr = accessors[nIdx];
                normals.resize(nAcsr.count);
                auto& nBufV = bufViews[nAcsr.bufferView];
                auto& nBuf = bufs[nBufV.buffer];
                byteOffset = nBufV.byteOffset +
                    nAcsr.byteOffset;
                const float* normData = reinterpret_cast<const float*>(&nBuf.data[byteOffset]);
                for (int n = 0; n < normals.size(); ++n) {
                    Normal3f norm(normData[3 * n], normData[3 * n + 1], normData[3 * n + 2]);
                    normals[n] = norm;
                }
            }

            if (hasTexCoord_0) {
                auto& texAcsr = accessors[tex_0_Idx];
                texCoords_0.resize(texAcsr.count);
                auto& tex_0_BufV = bufViews[texAcsr.bufferView];
                auto& text_0_Buf = bufs[tex_0_BufV.buffer];
                byteOffset = tex_0_BufV.byteOffset +
                    texAcsr.byteOffset;
            }
            

            meshes.push_back(std::make_shared<TriangleMesh>(
            ));
        }
    }

};
#endif