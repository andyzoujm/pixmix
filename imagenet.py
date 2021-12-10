
import argparse
import os
import random
import shutil
import time
import warnings
import math
import numpy as np
from PIL import ImageOps, Image
import tempfile

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

import pixmix_utils as utils
from collections import OrderedDict
from calibration_tools import *

utils.IMAGE_SIZE = 224

parser = argparse.ArgumentParser(description='ImageNet Training')
parser.add_argument('--data-standard', help='Path to dataset', default="data/imagenet/train/")
parser.add_argument('--data-val', help='Path to validation dataset', default="data/imagenet/val/")
parser.add_argument('--imagenet-r-dir', help='Path to ImageNet-R', default="data/imagenet_r/")
parser.add_argument('--imagenet-c-dir', help='Path to ImageNet-C', default="data/imagenet_c/")
parser.add_argument('--mixing-set', help='Path to mixing set', required=True)
parser.add_argument('--num-classes', choices=['200', '1000'], required=True)
parser.add_argument(
    '--aug-severity',
    default=1,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--beta',
    default=4,
    type=int,
    help='Severity of mixing')
parser.add_argument(
    '--k',
    default=4,
    type=int,
    help='Mixing iterations')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all augmentation operations (+brightness,contrast,color,sharpness).')
parser.add_argument('--save', default='checkpoints/TEMP', type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=36, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--batch-size-val', default=256, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set and Imagenet-C')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

args = parser.parse_args()
print(args)

##################################################
# ImageNet-1K classes
##################################################
all_classes = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811', 'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457', 'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244', 'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675', 'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143', 'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313', 'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672', 'n01860187', 'n01871265', 'n01872401', 'n01873310', 'n01877812', 'n01882714', 'n01883070', 'n01910747', 'n01914609', 'n01917289', 'n01924916', 'n01930112', 'n01943899', 'n01944390', 'n01945685', 'n01950731', 'n01955084', 'n01968897', 'n01978287', 'n01978455', 'n01980166', 'n01981276', 'n01983481', 'n01984695', 'n01985128', 'n01986214', 'n01990800', 'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849', 'n02013706', 'n02017213', 'n02018207', 'n02018795', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845', 'n02056570', 'n02058221', 'n02066245', 'n02071294', 'n02074367', 'n02077923', 'n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 'n02104365', 'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312', 'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341', 'n02110627', 'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137', 'n02112350', 'n02112706', 'n02113023', 'n02113186', 'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02114367', 'n02114548', 'n02114712', 'n02114855', 'n02115641', 'n02115913', 'n02116738', 'n02117135', 'n02119022', 'n02119789', 'n02120079', 'n02120505', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02129165', 'n02129604', 'n02130308', 'n02132136', 'n02133161', 'n02134084', 'n02134418', 'n02137549', 'n02138441', 'n02165105', 'n02165456', 'n02167151', 'n02168699', 'n02169497', 'n02172182', 'n02174001', 'n02177972', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338', 'n02236044', 'n02256656', 'n02259212', 'n02264363', 'n02268443', 'n02268853', 'n02276258', 'n02277742', 'n02279972', 'n02280649', 'n02281406', 'n02281787', 'n02317335', 'n02319095', 'n02321529', 'n02325366', 'n02326432', 'n02328150', 'n02342885', 'n02346627', 'n02356798', 'n02361337', 'n02363005', 'n02364673', 'n02389026', 'n02391049', 'n02395406', 'n02396427', 'n02397096', 'n02398521', 'n02403003', 'n02408429', 'n02410509', 'n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312', 'n02437616', 'n02441942', 'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366', 'n02454379', 'n02457408', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708', 'n02484975', 'n02486261', 'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166', 'n02490219', 'n02492035', 'n02492660', 'n02493509', 'n02493793', 'n02494079', 'n02497673', 'n02500267', 'n02504013', 'n02504458', 'n02509815', 'n02510455', 'n02514041', 'n02526121', 'n02536864', 'n02606052', 'n02607072', 'n02640242', 'n02641379', 'n02643566', 'n02655020', 'n02666196', 'n02667093', 'n02669723', 'n02672831', 'n02676566', 'n02687172', 'n02690373', 'n02692877', 'n02699494', 'n02701002', 'n02704792', 'n02708093', 'n02727426', 'n02730930', 'n02747177', 'n02749479', 'n02769748', 'n02776631', 'n02777292', 'n02782093', 'n02783161', 'n02786058', 'n02787622', 'n02788148', 'n02790996', 'n02791124', 'n02791270', 'n02793495', 'n02794156', 'n02795169', 'n02797295', 'n02799071', 'n02802426', 'n02804414', 'n02804610', 'n02807133', 'n02808304', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02817516', 'n02823428', 'n02823750', 'n02825657', 'n02834397', 'n02835271', 'n02837789', 'n02840245', 'n02841315', 'n02843684', 'n02859443', 'n02860847', 'n02865351', 'n02869837', 'n02870880', 'n02871525', 'n02877765', 'n02879718', 'n02883205', 'n02892201', 'n02892767', 'n02894605', 'n02895154', 'n02906734', 'n02909870', 'n02910353', 'n02916936', 'n02917067', 'n02927161', 'n02930766', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02951585', 'n02963159', 'n02965783', 'n02966193', 'n02966687', 'n02971356', 'n02974003', 'n02977058', 'n02978881', 'n02979186', 'n02980441', 'n02981792', 'n02988304', 'n02992211', 'n02992529', 'n02999410', 'n03000134', 'n03000247', 'n03000684', 'n03014705', 'n03016953', 'n03017168', 'n03018349', 'n03026506', 'n03028079', 'n03032252', 'n03041632', 'n03042490', 'n03045698', 'n03047690', 'n03062245', 'n03063599', 'n03063689', 'n03065424', 'n03075370', 'n03085013', 'n03089624', 'n03095699', 'n03100240', 'n03109150', 'n03110669', 'n03124043', 'n03124170', 'n03125729', 'n03126707', 'n03127747', 'n03127925', 'n03131574', 'n03133878', 'n03134739', 'n03141823', 'n03146219', 'n03160309', 'n03179701', 'n03180011', 'n03187595', 'n03188531', 'n03196217', 'n03197337', 'n03201208', 'n03207743', 'n03207941', 'n03208938', 'n03216828', 'n03218198', 'n03220513', 'n03223299', 'n03240683', 'n03249569', 'n03250847', 'n03255030', 'n03259280', 'n03271574', 'n03272010', 'n03272562', 'n03290653', 'n03291819', 'n03297495', 'n03314780', 'n03325584', 'n03337140', 'n03344393', 'n03345487', 'n03347037', 'n03355925', 'n03372029', 'n03376595', 'n03379051', 'n03384352', 'n03388043', 'n03388183', 'n03388549', 'n03393912', 'n03394916', 'n03400231', 'n03404251', 'n03417042', 'n03424325', 'n03425413', 'n03443371', 'n03444034', 'n03445777', 'n03445924', 'n03447447', 'n03447721', 'n03450230', 'n03452741', 'n03457902', 'n03459775', 'n03461385', 'n03467068', 'n03476684', 'n03476991', 'n03478589', 'n03481172', 'n03482405', 'n03483316', 'n03485407', 'n03485794', 'n03492542', 'n03494278', 'n03495258', 'n03496892', 'n03498962', 'n03527444', 'n03529860', 'n03530642', 'n03532672', 'n03534580', 'n03535780', 'n03538406', 'n03544143', 'n03584254', 'n03584829', 'n03590841', 'n03594734', 'n03594945', 'n03595614', 'n03598930', 'n03599486', 'n03602883', 'n03617480', 'n03623198', 'n03627232', 'n03630383', 'n03633091', 'n03637318', 'n03642806', 'n03649909', 'n03657121', 'n03658185', 'n03661043', 'n03662601', 'n03666591', 'n03670208', 'n03673027', 'n03676483', 'n03680355', 'n03690938', 'n03691459', 'n03692522', 'n03697007', 'n03706229', 'n03709823', 'n03710193', 'n03710637', 'n03710721', 'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03729826', 'n03733131', 'n03733281', 'n03733805', 'n03742115', 'n03743016', 'n03759954', 'n03761084', 'n03763968', 'n03764736', 'n03769881', 'n03770439', 'n03770679', 'n03773504', 'n03775071', 'n03775546', 'n03776460', 'n03777568', 'n03777754', 'n03781244', 'n03782006', 'n03785016', 'n03786901', 'n03787032', 'n03788195', 'n03788365', 'n03791053', 'n03792782', 'n03792972', 'n03793489', 'n03794056', 'n03796401', 'n03803284', 'n03804744', 'n03814639', 'n03814906', 'n03825788', 'n03832673', 'n03837869', 'n03838899', 'n03840681', 'n03841143', 'n03843555', 'n03854065', 'n03857828', 'n03866082', 'n03868242', 'n03868863', 'n03871628', 'n03873416', 'n03874293', 'n03874599', 'n03876231', 'n03877472', 'n03877845', 'n03884397', 'n03887697', 'n03888257', 'n03888605', 'n03891251', 'n03891332', 'n03895866', 'n03899768', 'n03902125', 'n03903868', 'n03908618', 'n03908714', 'n03916031', 'n03920288', 'n03924679', 'n03929660', 'n03929855', 'n03930313', 'n03930630', 'n03933933', 'n03935335', 'n03937543', 'n03938244', 'n03942813', 'n03944341', 'n03947888', 'n03950228', 'n03954731', 'n03956157', 'n03958227', 'n03961711', 'n03967562', 'n03970156', 'n03976467', 'n03976657', 'n03977966', 'n03980874', 'n03982430', 'n03983396', 'n03991062', 'n03992509', 'n03995372', 'n03998194', 'n04004767', 'n04005630', 'n04008634', 'n04009552', 'n04019541', 'n04023962', 'n04026417', 'n04033901', 'n04033995', 'n04037443', 'n04039381', 'n04040759', 'n04041544', 'n04044716', 'n04049303', 'n04065272', 'n04067472', 'n04069434', 'n04070727', 'n04074963', 'n04081281', 'n04086273', 'n04090263', 'n04099969', 'n04111531', 'n04116512', 'n04118538', 'n04118776', 'n04120489', 'n04125021', 'n04127249', 'n04131690', 'n04133789', 'n04136333', 'n04141076', 'n04141327', 'n04141975', 'n04146614', 'n04147183', 'n04149813', 'n04152593', 'n04153751', 'n04154565', 'n04162706', 'n04179913', 'n04192698', 'n04200800', 'n04201297', 'n04204238', 'n04204347', 'n04208210', 'n04209133', 'n04209239', 'n04228054', 'n04229816', 'n04235860', 'n04238763', 'n04239074', 'n04243546', 'n04251144', 'n04252077', 'n04252225', 'n04254120', 'n04254680', 'n04254777', 'n04258138', 'n04259630', 'n04263257', 'n04264628', 'n04265275', 'n04266014', 'n04270147', 'n04273569', 'n04275548', 'n04277352', 'n04285008', 'n04286575', 'n04296562', 'n04310018', 'n04311004', 'n04311174', 'n04317175', 'n04325704', 'n04326547', 'n04328186', 'n04330267', 'n04332243', 'n04335435', 'n04336792', 'n04344873', 'n04346328', 'n04347754', 'n04350905', 'n04355338', 'n04355933', 'n04356056', 'n04357314', 'n04366367', 'n04367480', 'n04370456', 'n04371430', 'n04371774', 'n04372370', 'n04376876', 'n04380533', 'n04389033', 'n04392985', 'n04398044', 'n04399382', 'n04404412', 'n04409515', 'n04417672', 'n04418357', 'n04423845', 'n04428191', 'n04429376', 'n04435653', 'n04442312', 'n04443257', 'n04447861', 'n04456115', 'n04458633', 'n04461696', 'n04462240', 'n04465501', 'n04467665', 'n04476259', 'n04479046', 'n04482393', 'n04483307', 'n04485082', 'n04486054', 'n04487081', 'n04487394', 'n04493381', 'n04501370', 'n04505470', 'n04507155', 'n04509417', 'n04515003', 'n04517823', 'n04522168', 'n04523525', 'n04525038', 'n04525305', 'n04532106', 'n04532670', 'n04536866', 'n04540053', 'n04542943', 'n04548280', 'n04548362', 'n04550184', 'n04552348', 'n04553703', 'n04554684', 'n04557648', 'n04560804', 'n04562935', 'n04579145', 'n04579432', 'n04584207', 'n04589890', 'n04590129', 'n04591157', 'n04591713', 'n04592741', 'n04596742', 'n04597913', 'n04599235', 'n04604644', 'n04606251', 'n04612504', 'n04613696', 'n06359193', 'n06596364', 'n06785654', 'n06794110', 'n06874185', 'n07248320', 'n07565083', 'n07579787', 'n07583066', 'n07584110', 'n07590611', 'n07613480', 'n07614500', 'n07615774', 'n07684084', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07711569', 'n07714571', 'n07714990', 'n07715103', 'n07716358', 'n07716906', 'n07717410', 'n07717556', 'n07718472', 'n07718747', 'n07720875', 'n07730033', 'n07734744', 'n07742313', 'n07745940', 'n07747607', 'n07749582', 'n07753113', 'n07753275', 'n07753592', 'n07754684', 'n07760859', 'n07768694', 'n07802026', 'n07831146', 'n07836838', 'n07860988', 'n07871810', 'n07873807', 'n07875152', 'n07880968', 'n07892512', 'n07920052', 'n07930864', 'n07932039', 'n09193705', 'n09229709', 'n09246464', 'n09256479', 'n09288635', 'n09332890', 'n09399592', 'n09421951', 'n09428293', 'n09468604', 'n09472597', 'n09835506', 'n10148035', 'n10565667', 'n11879895', 'n11939491', 'n12057211', 'n12144580', 'n12267677', 'n12620546', 'n12768682', 'n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670', 'n13054560', 'n13133613', 'n15075141']
classes_chosen_1000 = all_classes
assert len(classes_chosen_1000) == 1000

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

##################################################
# ImageNet-R classes
##################################################
imagenet_r_wnids = ['n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677']
imagenet_r_wnids.sort()
classes_chosen_200 = imagenet_r_wnids[:] # Choose 100 classes for our dataset
assert len(classes_chosen_200) == 200
imagenet_r_mask = [wnid in classes_chosen_200 for wnid in all_classes]

##################################################
# ImageNet-A classes
##################################################
imagenet_a_wnids = ['n03355925', 'n03255030', 'n02504458', 'n01847000', 'n01910747', 'n02037110', 'n12144580', 'n03388043', 'n01531178', 'n02883205', 'n04131690', 'n07697313', 'n02951358', 'n02190166', 'n04456115', 'n03840681', 'n04347754', 'n04310018', 'n02980441', 'n04208210', 'n02259212', 'n01580077', 'n04086273', 'n01774750', 'n03014705', 'n02701002', 'n03888257', 'n02174001', 'n02895154', 'n04606251', 'n03721384', 'n02280649', 'n02051845', 'n03891332', 'n03384352', 'n02177972', 'n07753592', 'n02281787', 'n04235860', 'n03584829', 'n02233338', 'n04146614', 'n12057211', 'n02007558', 'n02356798', 'n03250847', 'n04033901', 'n01698640', 'n02129165', 'n07831146', 'n07714990', 'n02690373', 'n03026506', 'n02085620', 'n04141076', 'n04509417', 'n03444034', 'n02325366', 'n01694178', 'n03854065', 'n04344873', 'n03617480', 'n04389033', 'n02793495', 'n02279972', 'n03788195', 'n01843383', 'n02730930', 'n04366367', 'n04118538', 'n02317335', 'n02165456', 'n02133161', 'n04591713', 'n04019541', 'n07749582', 'n04067472', 'n01944390', 'n02879718', 'n02486410', 'n03452741', 'n01498041', 'n01784675', 'n04317175', 'n02906734', 'n02106550', 'n01677366', 'n01986214', 'n02948072', 'n02672831', 'n01820546', 'n03804744', 'n03717622', 'n02454379', 'n07718472', 'n04507155', 'n01882714', 'n03724870', 'n01833805', 'n04275548', 'n04252225', 'n02802426', 'n04133789', 'n03935335', 'n03590841', 'n01914609', 'n03124043', 'n04099969', 'n04179913', 'n02119022', 'n07697537', 'n01735189', 'n04254120', 'n02676566', 'n02127052', 'n01687978', 'n03666591', 'n01770393', 'n02814860', 'n02077923', 'n04270147', 'n02236044', 'n01819313', 'n04355338', 'n02206856', 'n01534433', 'n04376876', 'n04147183', 'n04532670', 'n02777292', 'n02445715', 'n01631663', 'n04039381', 'n04540053', 'n03837869', 'n03187595', 'n04482393', 'n02106662', 'n01924916', 'n02782093', 'n03125729', 'n02669723', 'n02655020', 'n07760859', 'n02231487', 'n02837789', 'n02009912', 'n12267677', 'n09229709', 'n02346627', 'n01558993', 'n03982430', 'n02992211', 'n02797295', 'n02361337', 'n04252077', 'n03291819', 'n01641577', 'n01669191', 'n01614925', 'n02410509', 'n02123394', 'n07583066', 'n03720891', 'n02110958', 'n01855672', 'n07734744', 'n03594945', 'n02787622', 'n02999410', 'n09472597', 'n03775071', 'n02268443', 'n04399382', 'n07768694', 'n02492035', 'n11879895', 'n03445924', 'n07695742', 'n03443371', 'n09835506', 'n03325584', 'n03670208', 'n04442312', 'n01770081', 'n02815834', 'n02226429', 'n02219486', 'n03417042', 'n03196217', 'n04554684', 'n04562935', 'n01985128', 'n02099601', 'n09246464', 'n07720875', 'n03223299', 'n01616318', 'n03483316', 'n02137549']
imagenet_a_wnids.sort()
assert len(imagenet_a_wnids) == 200
imagenet_a_mask = [wnid in imagenet_a_wnids for wnid in all_classes]


if args.num_classes == '200':
    classes_chosen = classes_chosen_200
elif args.num_classes == '1000':
    classes_chosen = classes_chosen_1000
else:
    raise NotImplementedError

if os.path.exists(args.save):
    resp = "None"
    while resp.lower() not in {'y', 'n'}:
        resp = input("Save directory {0} exits. Continue? [Y/n]: ".format(args.save))
        if resp.lower() == 'y':
            break
        elif resp.lower() == 'n':
            exit(1)
        else:
            pass
else:
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)
    else:
        print("Made save directory", args.save)

class ImageNetSubsetDataset(datasets.ImageFolder):
    """
    Dataset class to take a specified subset of some larger dataset
    """
    def __init__(self, root, *args, **kwargs):
        
        # print(f"Using {len(classes_chosen)} classes")

        self.new_root = tempfile.mkdtemp()
        for _class in classes_chosen:
            orig_dir = os.path.join(root, _class)
            assert os.path.isdir(orig_dir)

            os.symlink(orig_dir, os.path.join(self.new_root, _class))
        
        super().__init__(self.new_root, *args, **kwargs)
    
    def __del__(self):
        # Clean up
        shutil.rmtree(self.new_root)


def pixmix(orig, mixing_pic, preprocess):
  
  mixings = utils.mixings
  tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
  if np.random.random() < 0.5:
    mixed = tensorize(augment_input(orig))
  else:
    mixed = tensorize(orig)
  
  for _ in range(np.random.randint(args.k + 1)):
    
    if np.random.random() < 0.5:
      aug_image_copy = tensorize(augment_input(orig))
    else:
      aug_image_copy = tensorize(mixing_pic)

    mixed_op = np.random.choice(mixings)
    mixed = mixed_op(mixed, aug_image_copy, args.beta)
    mixed = torch.clip(mixed, 0, 1)

  return normalize(mixed)

def augment_input(image):
  aug_list = utils.augmentations_all if args.all_ops else utils.augmentations
  op = np.random.choice(aug_list)
  return op(image.copy(), args.aug_severity)


class PixMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform PixMix."""

  def __init__(self, dataset, mixing_set, preprocess):
    self.dataset = dataset
    self.mixing_set = mixing_set
    self.preprocess = preprocess

  def __getitem__(self, i):
    x, y = self.dataset[i]
    rnd_idx = np.random.choice(len(self.mixing_set))
    mixing_pic, _ = self.mixing_set[rnd_idx]
    return pixmix(x, mixing_pic, self.preprocess), y

  def __len__(self):
    return len(self.dataset)

best_acc1 = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    main_worker(args.gpu, args)

def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'vgg16':
        model = models.vgg16(
            pretrained=args.pretrained, 
            use_deepaugment_realtime=True
        )
        model.classifier[-1] = torch.nn.Linear(4096, len(classes_chosen))
        print(model)
    elif args.arch == 'vgg11':
        model = models.vgg11(
            pretrained=args.pretrained, 
            use_deepaugment_realtime=True
        )
        model.classifier[-1] = torch.nn.Linear(4096, len(classes_chosen))
        print(model)
    elif args.arch == 'resnet18':
        model = models.resnet18(
            pretrained=args.pretrained
        )
        if len(classes_chosen) != 1000:
            model.fc = torch.nn.Linear(512, len(classes_chosen))
        print(model)
    elif args.arch == 'resnet50':
        model = models.resnet50(
            pretrained=args.pretrained
        )
        if len(classes_chosen) != 1000:
            model.fc = torch.nn.Linear(2048, len(classes_chosen))
        print(model)
    elif args.arch == 'deit':
        model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
        if len(classes_chosen) != 1000:
            model.head = torch.nn.Linear(192, len(classes_chosen), bias=True)
        print(model)
    else:
        raise NotImplementedError()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    if args.arch == 'deit':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            print('Start epoch:', args.start_epoch)
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if args.data_standard == None:
        return
    
    val_loader = torch.utils.data.DataLoader(
        ImageNetSubsetDataset(
            args.data_val, 
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        ),
        batch_size=args.batch_size_val, shuffle=False,
        num_workers=32, pin_memory=True)
    
    val_loader_imagenet_r = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.imagenet_r_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=32, pin_memory=True)

    # val_loader_imagenet_a = torch.utils.data.DataLoader(
    #     datasets.ImageFolder("/var/tmp/datasets/imagenet_a", transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=32, pin_memory=True)

    if args.evaluate:
        _, val_top1, val_top5 = validate(val_loader, model, criterion, args)
        _, r_top1, r_top5 = validate(val_loader_imagenet_r, model, criterion, args, r=True)
        # _, a_top1, a_top5 = validate(val_loader_imagenet_a, model, criterion, args, a=True)
        with open(os.path.join(args.save, f"eval_results.csv"), 'a') as f:
            f.write('val_top1,val_top5,r_top1,r_top5\n')
            f.write('%0.5f,%0.5f,%0.5f,%0.5f\n' % (
                val_top1, val_top5, r_top1, r_top5
            ))
        evaluate_c(model, normalize, args)
        # evaluate_c_bar(model, normalize, args)
        print('FINISHED EVALUATION')
        return

    ######################
    # Training
    ######################

    print(f"Using Dataset {args.data_standard}")
    train_data = ImageNetSubsetDataset(
        args.data_standard,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ])
    )
    
    mixing_set = datasets.ImageFolder(
        args.mixing_set, 
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224)
        ])
    )

    print('train_size', len(train_data))
    print('mixing_set_size', len(mixing_set))

    train_dataset = PixMixDataset(train_data, mixing_set, {'normalize': normalize, 'tensorize': to_tensor})

    # Fix dataloader worker issue
    # https://github.com/pytorch/pytorch/issues/5059
    def wif(id):
        uint64_seed = torch.initial_seed()
        ss = np.random.SeedSequence([uint64_seed])
        # More than 128 bits (4 32-bit words) would be overkill.
        np.random.seed(ss.generate_state(4))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, worker_init_fn=wif)
    
    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / (args.lr * args.batch_size / 256.)))

    if args.start_epoch != 0:
        scheduler.step(args.start_epoch * len(train_loader))

    
    ###########################################################################
    ##### Main Training Loop
    ###########################################################################

    if not args.resume:
        with open(os.path.join(args.save, 'training_log.csv'), 'w') as f:
            f.write('epoch,train_loss,train_acc1,train_acc5,val_loss,val_acc1,val_acc5,R_loss,R_acc1,R_acc5\n')

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_losses_avg, train_top1_avg, train_top5_avg = train(train_loader, model, criterion, optimizer, scheduler, epoch, args)
        
        print("Evaluating on validation set")
        val_losses_avg, val_top1_avg, val_top5_avg = validate(val_loader, model, criterion, args)

        val_R_losses_avg, val_R_top1_avg, val_R_top5_avg = validate(val_loader_imagenet_r, model, criterion, args, r=True)

        # Save results in log file
        with open(os.path.join(args.save, 'training_log.csv'), 'a') as f:
            f.write('%03d,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f\n' % (
                (epoch + 1),
                train_losses_avg, train_top1_avg, train_top5_avg,
                val_losses_avg, val_top1_avg, val_top5_avg,
                val_R_losses_avg, val_R_top1_avg, val_R_top5_avg
            ))

        # remember best acc@1 and save checkpoint
        is_best = val_top1_avg > best_acc1
        best_acc1 = max(val_top1_avg, best_acc1)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


    #########################
    # evaluate on C
    #########################

    evaluate_c(model, normalize, args)
    # evaluate_c_bar(model, normalize, args)

    
def evaluate_c(model, normalize, args):

    model.eval()

    args.data_path = args.imagenet_c_dir
    with open(os.path.join(args.save, f"eval_imagenet_c_results.csv"), 'w') as f:
        f.write('corruption,strength,top1_accuracy,calib\n')
    
    corruptions = [e for e in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, e))] # All subdirectories, ignoring normal files
    corruptions = list(reversed(sorted(corruptions)))
    # corruptions = np.array_split(corruptions, args.total_workers)[args.worker_number]
    accuracy = []
    calibs = []
    for corr in corruptions:
        for strength in {1,2,3,4,5}: # choose strengths
            data_path = os.path.join(args.data_path, corr, str(strength))

            dataset = ImageNetSubsetDataset(
                data_path,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
            )

            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=False,
                num_workers=args.workers, 
                pin_memory=True
            )

            # Eval on this dataset
            
            acc, test_confidence, test_correct = get_net_results(dataloader, model)
            print(f"Eval on {corr} with strength {strength}: {acc}")

            curr_calib = calib_err(test_confidence, test_correct, p='2')

            with open(os.path.join(args.save, f"eval_imagenet_c_results.csv"), 'a') as f:
                f.write('%s,%d,%0.5f,%0.5f\n' % (
                    corr,
                    strength,
                    acc,
                    curr_calib
                ))
            accuracy.append(acc)
            calibs.append(curr_calib)
            
            del dataset 
            del dataloader
    
    print("Accuracy on Imagenet-C: {:.3f}".format(100 * np.mean(accuracy)))
    print('RMS {:.3f}\n'.format(100 * np.mean(calibs)))


def evaluate_c_bar(model, normalize, args):

    model.eval()

    args.data_path = "/var/tmp/datasets/imagenet_c_bar"
    with open(os.path.join(args.save, f"eval_imagenet_c_bar_results.csv"), 'w') as f:
        f.write('corruption,strength,top1_accuracy,calib\n')
    
    corruptions = [e for e in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, e))] # All subdirectories, ignoring normal files
    corruptions = list(reversed(sorted(corruptions)))
    # corruptions = np.array_split(corruptions, args.total_workers)[args.worker_number]
    accuracy = []
    calibs = []
    for corr in corruptions:
        for strength in {1,2,3,4,5}: # choose strengths
            data_path = os.path.join(args.data_path, corr, str(strength))

            dataset = ImageNetSubsetDataset(
                data_path,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
            )

            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=False,
                num_workers=args.workers, 
                pin_memory=True
            )

            # Eval on this dataset
            
            acc, test_confidence, test_correct = get_net_results(dataloader, model)
            print(f"Eval on {corr} with strength {strength}: {acc}")

            curr_calib = calib_err(test_confidence, test_correct, p='2')

            with open(os.path.join(args.save, f"eval_imagenet_c_bar_results.csv"), 'a') as f:
                f.write('%s,%d,%0.5f,%0.5f\n' % (
                    corr,
                    strength,
                    acc,
                    curr_calib
                ))
            accuracy.append(acc)
            calibs.append(curr_calib)
            
            del dataset 
            del dataloader
    
    print("Accuracy on Imagenet-C-Bar: {:.3f}".format(100 * np.mean(accuracy)))
    print('RMS {:.3f}\n'.format(100 * np.mean(calibs)))


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        bx = images.cuda(args.gpu, non_blocking=True)
        by = target.cuda(args.gpu, non_blocking=True)
        
        logits = model(bx)
        loss = criterion(logits, by)
        output, target = logits, by 

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args, r=False, a=False, adv=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    if adv:
        print('EVALUATING AGAINST ADVERSARY')
    elif r:
        print('EVALUATING ON IMAGENET-R')
    elif a:
        if args.num_classes == '200': # the 200 classes are different
            return 0,0,0
        print('EVALUATING ON IMAGENET-A')

    to_np = lambda x: x.data.to('cpu').numpy()

    confidence = []
    correct = []

    num_correct = 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # adversarial
            if adv:
                images = adv(model, images, target)

            # compute output
            output = model(images)
            if r and args.num_classes == '1000': # eval on ImangeNet-R
                output = output[:,imagenet_r_mask]
            elif a:
                output = output[:,imagenet_a_mask]

            loss = criterion(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            num_correct += pred.eq(target.data).sum().item()

            confidence.extend(to_np(F.softmax(output, dim=1).max(1)[0]).squeeze().tolist())
            pred = output.data.max(1)[1]
            correct.extend(pred.eq(target).to('cpu').numpy().squeeze().tolist())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    
    print('RMS {:.3f}\n'.format(100 * calib_err(np.array(confidence.copy()), np.array(correct.copy()), p='2')))

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, filename=os.path.join(args.save, "model.pth.tar")):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, './model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_net_results(dataloader, net):
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.to('cpu').numpy()

    confidence = []
    correct = []

    num_correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()

            output = net(data)

            # accuracy
            pred = output.data.max(1)[1]
            num_correct += pred.eq(target.data).sum().item()

            confidence.extend(to_np(F.softmax(output, dim=1).max(1)[0]).squeeze().tolist())
            pred = output.data.max(1)[1]
            correct.extend(pred.eq(target).to('cpu').numpy().squeeze().tolist())

    return num_correct / len(dataloader.dataset), np.array(confidence.copy()), np.array(correct.copy())

if __name__ == '__main__':
    main()
