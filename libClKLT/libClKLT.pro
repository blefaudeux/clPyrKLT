#-------------------------------------------------
#
# Project created by QtCreator 2014-01-10T16:55:44
#
#-------------------------------------------------

QT       -= core gui

TARGET = libClKLT

TEMPLATE = lib

DEFINES += LIBCLKLT_LIBRARY

SOURCES += \
    src/kernel_code.cl \
    src/clklt.cpp

HEADERS += \
    headers/clklt.h

INCLUDEPATH +=headers

QMAKE_CXXFLAGS += -std=c++0x

LIBS+= -lOpenCL

unix:!symbian {
    maemo5 {
        target.path = /opt/usr/lib
    } else {
        target.path = /usr/lib
    }
    INSTALLS += target
}
