# Assignment 2: Once you're done with images, then take short live video sequence and detect faces.
import cv2 as cv
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
import mtcnn
print(mtcnn.__version__)


def draw_faces(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot each face as a subplot
    for i in range(len(result_list)):
        # get coordinates
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        # define subplot
        pyplot.subplot(1, len(result_list), i+1)
        pyplot.axis('off')
        # plot face
        pyplot.imshow(data[y1:y2, x1:x2])
    # show the plot
    pyplot.show()


def draw_image_with_boxes(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    # show the plot
    pyplot.show()


cap = cv.VideoCapture(0)
detector = MTCNN()
while True:

    # Read the next frame from the video
    ret, frame = cap.read()

    # Break the loop if there are no more frames to read
    if not ret:
        break

        # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces using MTCNN
    results = detector.detect_faces(frame)

    # Draw bounding boxes around the faces
    for result in results:
        x, y, width, height = result['box']
        cv.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)

        # Display the frame
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('z'):
        break


# Release the resources
cap.release()
cv.destroyAllWindows()
