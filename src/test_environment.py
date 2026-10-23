import sys
import importlib

# Lista de pacotes necessários para o projeto
REQUIRED_PACKAGES = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("rtmlib", "rtmlib"),
    ("onnxruntime", "onnxruntime")
]

def check_dependencies():
    print("=" * 60)
    print("      SPIKE.AI - DIAGNÓSTICO DE DEPENDÊNCIAS E AMBIENTE")
    print("=" * 60)
    print(f"Versão do Python: {sys.version.split()[0]}")
    print("-" * 60)

    all_passed = True

    # 1. Teste de Importação das Bibliotecas
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "Versão desconhecida")
            print(f"[OK]  {pip_name:<20} -> Instalado (v{version})")
        except ImportError:
            print(f"[ERRO] {pip_name:<20} -> NÃO ENCONTRADO!")
            all_passed = False

    print("-" * 60)

    # 2. Teste de Suporte a GPU / CUDA (PyTorch)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[INFO] GPU disponível via CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] GPU não disponível / CUDA desativado. O código rodará em CPU.")
    except Exception:
        pass

    print("-" * 60)

    # 3. Teste de Conexão com o PyTorch Hub (Necessário para o MiDaS Depth)
    try:
        import torch
        print("Testando download/carregamento leve do modelo de profundidade (MiDaS)...")
        _ = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False)
        print("[OK]  Acesso ao PyTorch Hub para MiDaS funcionando perfeitamente!")
    except Exception as e:
        print(f"[AVISO] Falha ao testar PyTorch Hub/MiDaS: {e}")
        print("       (Certifique-se de ter conexão com a internet no primeiro uso).")

    print("=" * 60)
    if all_passed:
        print(">>> STATUS: Tudo pronto! Todas as dependências estão instaladas. <<<")
    else:
        print(">>> STATUS: Algumas dependências estão ausentes. Veja abaixo como instalar. <<<")
    print("=" * 60)

if __name__ == "__main__":
    check_dependencies()