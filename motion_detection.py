import cv2

static_back = None

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FPS, 2)
fps = int(vid.get(5))
print(f"fps: {fps}")

while True:
    check, frame = vid.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_gaussian = cv2.GaussianBlur(gray, (21, 21), 0)

    if static_back is None:
        static_back = gray_gaussian

    diff_frame = cv2.absdiff(static_back, gray_gaussian)

    thresh_frame = cv2.threshold(diff_frame, 50, 255, cv2.THRESH_BINARY)[1]
    thresh_frame_dilated = cv2.dilate(thresh_frame, None, iterations = 2)

    cv2.imshow('frame', frame)

    cv2.imshow('gaussian', gray_gaussian)

    cv2.imshow('thresh', thresh_frame)

    cv2.imshow('dilated', thresh_frame_dilated)

    if cv2.waitKey(1) == ord('q'):
        break

vid.release()

cv2.destroyAllWindows()