//---------------------------------------------------------------------------

#include <vcl.h>
#include <onnxruntime_cxx_api.h>
#pragma hdrstop
//---------------------------------------------------------------------------
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

USEFORM("..\appcmn\refdlg.cpp", RefDialog);
USEFORM("..\appcmn\timedlg.cpp", TimeDialog);
USEFORM("..\appcmn\viewer.cpp", TextViewer);
USEFORM("..\appcmn\vieweropt.cpp", ViewerOptDialog);
USEFORM("kmzconv.cpp", ConvDialog);
USEFORM("..\appcmn\aboutdlg.cpp", AboutDialog);
USEFORM("..\appcmn\confdlg.cpp", ConfDialog);
USEFORM("..\appcmn\freqdlg.cpp", FreqDialog);
USEFORM("..\appcmn\keydlg.cpp", KeyDialog);
USEFORM("..\appcmn\maskoptdlg.cpp", MaskOptDialog);
USEFORM("postmain.cpp", MainForm);
USEFORM("postopt.cpp", OptDialog);
//---------------------------------------------------------------------------
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
	try
	{
		Application->Initialize();
		test_onnx_link();
		Application->Title = "RTKPOST";
		Application->CreateForm(__classid(TMainForm), &MainForm);
		Application->CreateForm(__classid(TOptDialog), &OptDialog);
		Application->CreateForm(__classid(TConvDialog), &ConvDialog);
		Application->CreateForm(__classid(TOptDialog), &OptDialog);
		Application->CreateForm(__classid(TTextViewer), &TextViewer);
		Application->CreateForm(__classid(TViewerOptDialog), &ViewerOptDialog);
		Application->CreateForm(__classid(TRefDialog), &RefDialog);
		Application->CreateForm(__classid(TTimeDialog), &TimeDialog);
		Application->CreateForm(__classid(TConfDialog), &ConfDialog);
		Application->CreateForm(__classid(TAboutDialog), &AboutDialog);
		Application->CreateForm(__classid(TKeyDialog), &KeyDialog);
		Application->CreateForm(__classid(TMaskOptDialog), &MaskOptDialog);
		Application->CreateForm(__classid(TFreqDialog), &FreqDialog);
		Application->Run();
	}
	catch (Exception &exception)
	{
		Application->ShowException(&exception);
	}
	catch (...)
	{
		try
		{
			throw Exception("");
		}
		catch (Exception &exception)
		{
			Application->ShowException(&exception);
		}
	}
	return 0;
}
//---------------------------------------------------------------------------
