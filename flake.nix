{
  description = "EmbeddingGemma 2 direct-KFD backend, no ROCm runtime";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/b4fd65b198c599cbe814fcb9f42d25d021595ec9";
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      llvm = pkgs.llvmPackages_21;
      native = [ pkgs.cmake pkgs.ninja pkgs.pkg-config llvm.lld ];
      package = llvm.stdenv.mkDerivation {
        pname = "embeddinggemma-kfd";
        version = "0.1.0";
        src = self;
        nativeBuildInputs = native;
        buildInputs = [ pkgs.libdrm ];
        cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release"
          "-DLIBKFD_GPU_COMPILER=${llvm.clang-unwrapped}/bin/clang"
          "-DLIBKFD_GPU_RESOURCE_DIR=${llvm.clang}/resource-root" ];
        # AMDGPU assembly/code objects are not host ELF files.
        dontStrip = true;
        # Only CPU regressions are registered with CTest. GPU tests stay manual.
        doCheck = true;
      };
    in {
      packages.${system}.default = package;
      devShells.${system}.default = llvm.stdenv.mkDerivation {
        name = "embeddinggemma-kfd-dev";
        shellHook = ''
          export KFD_GPU_COMPILER=${llvm.clang-unwrapped}/bin/clang
          export KFD_GPU_RESOURCE_DIR=${llvm.clang}/resource-root
        '';
        nativeBuildInputs = native ++ [ pkgs.python3 pkgs.git pkgs.curl pkgs.gitleaks ];
        buildInputs = [ pkgs.libdrm ];
      };
    };
}
