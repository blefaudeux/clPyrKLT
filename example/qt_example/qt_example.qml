import QtQuick 1.1

Rectangle {
    width: 360
    height: 360
    Text {
        anchors.centerIn: parent
        text: "Initial Optical Flow demo"
    }
    MouseArea {
        anchors.fill: parent
        onClicked: {
            Qt.quit();
        }
    }
}

