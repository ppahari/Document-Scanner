import cv2
import numpy as np
import urllib2
import os.path
import PIL
import PIL.Image

def resize(image, height = 600):
    """
    :param image: Given Image matrix
    :param height: Given height, if parameter not given take 600 as default
    :return: Resized Image
    """
    # Row is the height of the image and column is the width of the image
    # New Height to original image height ratio (aspect  ratio)
    ratio = float(height)/float(image.shape[0])
    # dimension - width * ratio , given height
    dimension = (int(image.shape[1]*ratio), int(height))
    res = cv2.resize(image, dimension)
    return res


def detectEdge(image):
    """
    Detects the edges of input image matrix
    :param image: Image matrix
    :return: edged Image matrix
    """
    # Convert Image into Grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bilateral filter is highly effective in reducing noise and keeping the edges sharp , but very slow
    # Parameter used - src, d- Diameter of each pixel neighborhood that is used during filtering
    # & sigmaColor = 75 , & sigma space =75 , if lessthan 10 - no effect, if >150 - too strong
    blur = cv2.bilateralFilter(img,9,75,75)
    # # Apply otsu thresholding to get optimal thresholding
    # ret, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edged = cv2.Canny(blur, 75, 200)
    return edged


def getPageContours(image):
    """
    Gets the 4 contour points of rectangle paper
    :param image: Image Matrix
    :return: 4 Pts of contour
    """
    # For Better accuracy we prefer canny edged/thresholded image here to detect contour
    img, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sorts the contour in increasing order based on their area, largest area contour first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        # Contour Approximation - Removes all the curves and approximates the contour to straight line
        perimeter = cv2.arcLength(cnt,True)
        # Epsilon is the maximum distance from contour to approximated contour
        epsilon = 0.05 * perimeter
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        l = len(approx)
        # If the largest area contour has 4 points, then we have got our page contour
        if l == 4:
            page_cnt = approx
            break

    return page_cnt




def euclideanDist(pt1,pt2):
    """
    Finds the euclidean distance between two points pt1 and pt2
    :param pt1: Point 1
    :param pt2: Point 2
    :return: distance between Point 1 (x1,y1) and Point 2 (x2,y2)
    """
    dist = np.sqrt(np.sum((pt1-pt2)**2))
    return dist


def perspectiveTransform(page_cnt, image):
    """
    Finds the 4 Starting Point and 4 Target point and perfoms the perspective transformation
    :param page_cnt: contours
    :param image: source image
    :return: transformed image
    """
    # Sum of x,y and diff of x,y
    # Points with Max sum is Top Right, Point with min sum is bottom left
    # Points with min diff is bottom right and point with max difference is top left
    c = page_cnt
    sum = np.sum(c,axis=1)
    diff = np.diff(c,axis=1)
    tl = c[np.argmax(diff)]
    tr = c[np.argmax(sum)]
    bl = c[np.argmin(sum)]
    br = c[np.argmin(diff)]
    sourcePoint = np.array([tl, tr, bl, br],np.float32)

    # Width is the distance between top left and top right also distance between bottom right and bottom left
    # Height is the distance between top left and bottom left also distance between top right and bottom right
    # We calculate distance between two points using Euclidean Distance

    hl = euclideanDist(tl, bl)  # hl -> Height between top left and bottom left
    hr = euclideanDist(tr, br)  # hr -> Height between top right and bottom right
    height = max(hl, hr)        # Finds the max height between 'hl' and 'hr'
    wt = euclideanDist(tl, tr)   # wt -> Width between top left and top right
    wb = euclideanDist(bl, br)   # wb -> Width between bottom left and bottom right
    width = max(wt, wb)         # Finds the max width between 'wt' and 'wb'

    new_tl = [0, height]
    new_tr = [width, height]
    new_bl = [0, 0]
    new_br = [width, 0]

    # Initalize the destination point
    dstPoint = np.array([new_tl, new_tr, new_bl, new_br], np.float32)
    # Finds the 3 x 3 transformation matrix with startPoint and dstPoint
    m = cv2.getPerspectiveTransform(sourcePoint, dstPoint)
    # Applies 3 x 3 transformation matrix to the image
    img = cv2.warpPerspective(image, m, (int(width), int(height)))
    return img

def captureWebcam():
    """
    This function captures the image from Ip Webcam
    :return: returns the captured Image from Ip Webcam stream
    """
    # Get the streaming address of ip webcam
    address = raw_input("Enter ip address: ")
    address += ":8080"
    streamurl = 'http://' + address + '/video'
    print 'Streaming ' + streamurl
    while True:
        try:
            stream = urllib2.urlopen(streamurl) # Opens the Streaming Address
            break
        except IOError:
            pass
    buffer=''
    i=0
    while True:
        buffer += stream.read(1024) # Buffer
        x = buffer.find('\xff\xd8') # Finds the beginning of jpg frame
        y = buffer.find('\xff\xd9') # Finds the end of jpg frame
        if x != -1 and y != -1:
            i += 1
            print 'Frame: %d\n' % i
            print '---Pre Buffer---'
            print buffer
            jpgframe = buffer[x:y+2]     # Only Grab the jpg frame from http stream
            buffer = buffer[y+2:]    # Bytes will be shortened to find another starting point of jpg frame
            # Decodes the string image from string to numpy array format
            print '---JPG FRAME---'
            print jpgframe
            print '---Post Buffer---'
            print buffer
            img = cv2.imdecode(np.fromstring(jpgframe, dtype=np.uint8), -1)
            cv2.namedWindow(streamurl, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(streamurl,600,750)
            cv2.imshow(streamurl, img)
            k = cv2.waitKey(1)
            if k == ord('c'):
                cv2.imwrite('CapturedImage.jpg',img)
                cv2.destroyAllWindows()
                return img
            elif k == 27:
                cv2.destroyAllWindows()
                exit(0)


def display(winname,image):
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname,500,650)
    cv2.imshow(winname, image)
    key = cv2.waitKey(0)
    if key == 27:    # This will destroy the respective window without saving it
        cv2.destroyWindow(winname)
    elif key==ord('p'):      # This choice will save image as jpg and as well as pdf
        fname = winname+'.jpg'
        cv2.imwrite(fname,image)
        print winname + 'Image saved as  ' + fname
        img = PIL.Image.open(fname)
        fname = winname+'.pdf'
        PIL.Image.Image.save(img,fname, "PDF", resoultion = 100.0)
        print winname + 'Image saved as  ' + fname
        cv2.destroyWindow(winname)
    elif key == ord('s'):   # This choice will save image as jpg
        fname=winname+'.jpg'
        cv2.imwrite(fname,image)
        print winname + 'Image saved as  ' + fname
        cv2.destroyWindow(winname)


def main():
    while True:
        print 'Enter 1 to enter image from disk'
        print 'Enter 2 to capture image from ip webcam'
        ip = raw_input('Enter Choice: ')
        if ip == '1':
            # Main Source Image
            while True:
                input = raw_input('Enter The Source Image File Name: ')
                status = os.path.exists(input)
                srcImage = cv2.imread(input)
                if status:
                    break
            break
        elif ip == '2':
            srcImage = captureWebcam()
            if srcImage.size:
                break
        else:
            pass
    # Read Image
    # Resize the image to the desired size
    resizedImage = resize(srcImage)
    print "Original Image Resized"
    # Detect the edge in the resized image - applies canny edge detection
    edged = detectEdge(resizedImage)
    print ' Edge Detected'
    # Detects the Page Contours in the image
    contours = getPageContours(edged)
    print '\n---Page Contours----\n'
    print contours
    # Draw Contour outline
    cim = cv2.drawContours(resizedImage.copy(),[contours],-1,(255,0,0),3)
    # Scale the page contours to source Image
    cnt = contours.reshape(4,2)*(srcImage.shape[0]/600.0)
    transformed = perspectiveTransform(cnt,srcImage)
    print "Image Transformed"
    # Gray the image
    grayed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's Thresholding to Transformed Image
    ret, th = cv2.threshold(grayed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print 'Image Thresholded'
    display('Original',srcImage)
    display('Edged',edged)
    display('Contour',cim)
    display('Scanned',transformed)
    display('ScannedThresholded',th)
if __name__ == "__main__":
    main()
