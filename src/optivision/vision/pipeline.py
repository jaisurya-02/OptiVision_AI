import cv2

from optivision.vision.detector import ObjectDetector


print("🚀 OptiVision AI pipeline started")


def main():

    detector = ObjectDetector()

    cap = cv2.VideoCapture(0)


    if not cap.isOpened():
        print("Camera not detected")
        return


    print("Camera started successfully")


    while True:

        success, frame = cap.read()


        if not success:
            print("Frame not received")
            break


        results = detector.detect(frame)


        output = results[0].plot()


        cv2.imshow(
            "OptiVision AI",
            output
        )


        if cv2.waitKey(1) == 27:
            break


    cap.release()

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()