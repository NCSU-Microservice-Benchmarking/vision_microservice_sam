from ultralytics import SAM
from PIL import Image
import cv2

# Load a model
model = SAM('sam_b.pt')

# Display model information (optional)
model.info()

# # Run inference and save as an image
results = model('bus.png')
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
# save results as a PNG image with opencv
cv2.imwrite('sam_results.png', im_array)
    # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    # im.save('sam_results.jpg')  # save image