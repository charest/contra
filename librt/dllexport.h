#ifndef LIBRT_DLLEXPORT_H
#define LIBRT_DLLEXPORT_H

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif


#endif // LIBRT_DLLEXPORT_H
