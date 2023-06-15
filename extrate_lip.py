import cv2

# Load the cascade
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the video
cap = cv2.VideoCapture('section_1_000.80_002.91.mp4')

while True:
    # Read a frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 3)

    for (x, y, w, h) in mouth_rects:
        y = int(y - 0.15 * h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        break



    # # Detect faces and mouths
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # for (x, y, w, h) in faces:
    #     roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = frame[y:y+h, x:x+w]
    #     mouths = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)
    #     for (mx, my, mw, mh) in mouths:
    #         cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)
    #         # Save the mouth image
    #         mouth_image = roi_color[my:my+mh, mx:mx+mw]
    #         cv2.imwrite('mouth.jpg', mouth_image)

    # Display the frame
    cv2.imshow('frame', frame)
    
    # Exit if ESC pressed
    if cv2.waitKey(1) == 27:
        break

# Release the capture
# cap.release()
# cv2.destroyAllWindows()
