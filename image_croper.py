from PIL import Image

x1, y1, x2, y2 = 0, 1261, 1080, 1477

img = Image.open("testcase/image_match/screens/xhs/xhs_target_13.png")
crop = img.crop((x1, y1, x2, y2))
crop.save("testcase/image_match/screens/xhs/xhs_template_13.png")