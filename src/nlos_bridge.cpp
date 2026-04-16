#include <vcl.h>
#include <onnxruntime_cxx_api.h>

extern "C" void test_onnx_link(void)
{
    try {
        const char *version = OrtGetApiBase()->GetVersionString();
        ShowMessage("ONNX Runtime caricato correttamente! Versione: " + String(version));
    }
    catch (...) {
        ShowMessage("Errore critico nel caricamento di ONNX Runtime!");
    }
}

