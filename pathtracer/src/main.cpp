#include<iostream>
#include<string>
#include<fstream>

int main() {

	std::string path = "D:\\3D Models\\sponza_gltf\\scene.gltf";
	std::ifstream file(path.c_str());
	std::cout<< file.good();
	return 1;
}