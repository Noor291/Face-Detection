# Assignment 1: Detect faces from live images taken from webcam.
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


key = cv. waitKey(1)
cam = cv.VideoCapture(0)
while cam.isOpened():
    try:
        check, frame = cam.read()
        # print(check) #prints true as long as the webcam is running
        # print(frame) #prints matrix values of each framecd
        cv.imshow("Capturing", frame)
        key = cv.waitKey(1)
        if key == ord('z'):
            cv.imwrite(filename='saved_img.jpg', img=frame)

            cam.release()

    except (KeyboardInterrupt):
        print("Turning off camera.")
        cam.release()
        print("Camera off.")
        print("Program ended.")
        cv.destroyAllWindows()
        break

filename = 'saved_img.jpg'
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image
draw_faces(filename, faces)
draw_image_with_boxes(filename, faces)
