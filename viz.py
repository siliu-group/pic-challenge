from utils import *

#load relation annotation
gtRel = Relation()
gtRel.loadAnnot('demo_data/pic/relations.json')

gtSemanticPath = 'demo_data/pic/semantic'
gtInstancePath = 'demo_data/pic/instance'

imgname = 'indoor_02018.jpg'

#load segmentation annotation
semanticImg = cv2.imread(os.path.join(gtSemanticPath,imgname.replace('.jpg','.png')), cv2.IMREAD_GRAYSCALE)
insImg = cv2.imread(os.path.join(gtInstancePath,imgname.replace('.jpg','.png')), cv2.IMREAD_GRAYSCALE)
img = cv2.imread(os.path.join('demo_data/pic/images',imgname))

#visualize segmentation
visualize_segmentation(semanticImg, img)

#visualize instance
visualize_instance(insImg, semanticImg, img)

relation = gtRel.loadRel(imgname)
#visualize relation
visualize_relation(relation,insImg,img)

#evalute code is coming soon
