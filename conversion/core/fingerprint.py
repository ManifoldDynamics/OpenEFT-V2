import os
from subprocess import check_output
import cv2
import math
from conversion.core.eft_helper import US_CHAR
from conversion.core.fd258_ocr import OCR_LOCATIONS

class Finger:
    def __init__(self, str):
        """
        Center (sx,sy)
        Dimensions (sw,sh)
        Angle (t)

        self.start
        self.end
        """
        self.orgID = "15" # Vendor ID for NFIQv1
        self.algID = "14205" # Algorithm ID for NFIQv1
        self.str = str
        self.readString()
        self.computeBox()
        self.segmentQuality()

    def readString(self):
        vals = self.str.split(' ')
        #[FILE, lfour_10.raw, ->, e, 3, sw, 168, sh, 280, sx, 120, sy, 256, th, -28.3]
        self.name = vals[1]
        print(vals)
        print(self.name, self.name.split('_'))
        self.n = str(int(self.name.split('_')[2].split('.')[0]))
        self.sw = int(vals[6])
        self.sh = int(vals[8])
        self.sx = int(vals[10])
        self.sy = int(vals[12])
        self.t = float(vals[14])

    def computeBox(self):
        self.x1 = str(abs(int((self.sw / 2) * math.cos(self.t) - (self.sh / 2)))) # Left
        self.y1 = str(abs(int((self.sh / 2) * math.cos(self.t) - (self.sw / 2)))) # Top
        self.x2 = str(abs(int((self.sw / 2) * math.cos(self.t) + (self.sh / 2)))) # Top
        self.y2 = str(abs(int((self.sh / 2) * math.cos(self.t) - (self.sw / 2)))) # Bottom
        
    def segmentQuality(self):
        # Four fields in each finger

        # finger number (1-10)
        # estimated correctends (254)
        # Vendor quality id
        # numeric product code 
        x = "nfiq {}".format(self.name)
        self.score = check_output(x, shell=True, text=True).strip()

    def getScoreString(self):
        if int(self.n) == 11:
            a = "1"
        elif int(self.n) == 12:
            a="6"
        else:
            a=self.n
        x = a + chr(US_CHAR) + self.score + chr(US_CHAR) + self.orgID + chr(US_CHAR) + self.algID
        return x

    def getPosString(self):
        x = self.n + chr(US_CHAR) + self.x1 + chr(US_CHAR) + self.x2 + chr(US_CHAR) + self.y1 + chr(US_CHAR) + self.y2
        return x

class Fingerprint:
    """
        self.tmpdir # The temporary directory files are stored in
        self.name # The file name (without encoding)
        self.encoding # The base encoding
        self.converted # The compressed fingerprint file
        self.src # The CV2 image used to extract fingerprint
        self.img # The CV2 image of the extracted fingerprint
        self.ppi # Pixel density
        self.hll # HORIZONTAL LINE LENGTH
        self.vll # VERTICAL LINE LENGTH
        self.hps # HORIZONTAL PIXEL SCALE
        self.vps # VERTICAL PIXEL SCALE
        self.slc # SCALE UNITS
        self.cga # COMPRESSION ALGORITHM (JP2 default)
        self.bpx # BITS PER PIXEL
        self.fgp # Fingerprint position
    """
    def __init__(self, loc, src_img, tmpdir):
        self.tmpdir = tmpdir
        self.name = loc.id
        self.fgp = loc.fp_number
        self.encoding = 'png'
        self.converted = ""
        self.src = src_img
        self.extract_fp(loc.bbox) # Sets self.img 
        self.fingers = []

    def get_settings(self):
        self.ppi = self.get_ppi()
        self.hll = self.img.shape[0]# HORIZONTAL LINE LENGTH
        self.vll = self.img.shape[1] # VERTICAL LINE LENGTH
        self.slc = "1" # SCALE UNITS
        self.bpx = 8 # BITS PER PIXEL, should be 8

    def get_ppi(self):
        # Image should be 8x8"
        x = self.src.shape[0] / 8
        y = self.src.shape[1] / 8
        # Real DPI
        # Now to convert image to correct DPI
        self.hps = round(x) # HORIZONTAL PIXEL SCALE
        self.vps = round(y) # VERTICAL PIXEL SCALE
        self.ppi = round((x+y)/2) 

    def extract_fp(self, bbox):
        (x,y,w,h) = bbox
        self.img = self.src[y:y+h, x:x+w]
#        self.img = cv2.resize(self.img, (1000,1000), interpolation=cv2.INTER_LINEAR_EXACT)
        self.save_image()

    def segment(self):
        # Only use nfseg for slaps/four fingers (fgp 13, 14) and 2 thumbs (15)
        if self.fgp in [13, 14, 15]:
            x=""
            if 'nt' in os.name:
                x += "wsl "
            x += "nfseg {} 1 1 1 0 {}".format(self.fgp, self.converted)
            try:
                a = check_output(x, shell=True, text=True).split('\n')
                for each in a:
                    tmp = each.split('FILE')
                    if len(tmp) > 1 and len(tmp[1])>1:
                        tmp = tmp[1]
                        self.fingers.append(Finger(tmp))
            except Exception as e:
                print(f"Error segmenting {self.fgp}: {e}")
        else:
            # Individual rolled prints are already single fingers, no need to use nfseg
            # We create a pseudo-nfseg string so Finger parses it correctly.
            # Format expected by Finger.readString:
            # [FILE, {name}, ->, e, 3, sw, {w}, sh, {h}, sx, {w/2}, sy, {h/2}, th, 0.0]
            # We'll make up a name that Finger expects: "indiv_finger_{fgp}.wsq"
            name = os.path.basename(self.converted)
            # The name split expects something like `prefix_type_num.ext`
            # so we'll construct: dummy_type_{self.fgp}.ext
            dummy_name = f"indiv_type_{self.fgp}.wsq"

            # Since we just pass the compressed file as the individual finger,
            # we want Finger.name to point to self.converted. But since name is parsed
            # from vals[1], we inject `self.converted` or just rename if needed.
            # Actually, the file itself is already at self.converted.
            # Let's see what Finger expects: it runs nfiq on vals[1], so vals[1] must be
            # the path to the file.

            w = self.img.shape[1]
            h = self.img.shape[0]
            sx = int(w/2)
            sy = int(h/2)

            # Make sure vals[1] is just the base name if we are in TMP_DIR
            # Actually `self.converted` is absolute path or relative to tmpdir?
            # It's `os.path.join(self.tmpdir, ...)`

            # Since `cwsq` creates the file, let's copy or rename it to dummy_name
            # so `Finger` can parse `_{self.fgp}.` correctly.
            dummy_path = os.path.join(self.tmpdir, dummy_name)
            import shutil
            shutil.copyfile(self.converted, dummy_path)

            dummy_str = f"FILE {dummy_name} -> e 3 sw {w} sh {h} sx {sx} sy {sy} th 0.0"
            self.fingers.append(Finger(dummy_str))



    def save_image(self, encoding=None):
        if encoding == None:
            f = os.path.join(self.tmpdir, self.name) + '.' + self.encoding
        else:
            f = os.path.join(self.tmpdir, self.name) + '.' + encoding
        cv2.imwrite(f, self.img)

    def convert(self, encoding='jp2', ratio=10):
        self.get_settings()
        i = os.path.join(self.tmpdir,self.name)
        o = i + '.' + encoding
        i = i + '.' + self.encoding
        x = ""
        if 'nt' in os.name:
            x += "wsl "
        if encoding == 'jp2':
            self.cga = "JP2" # COMPRESSION ALGORITHM [242-HQ-A6687913-SYSDOCU 3.82 value (ASCII)]
            x += "opj_compress -i {} -o {} -r {} -n 2".format(i,o, ratio)
        elif encoding == 'wsq':
            self.cga = "WSQ20"

            raw_file = os.path.join(self.tmpdir, self.name) + '.raw'
            with open(raw_file, 'wb') as f:
                f.write(self.img.tobytes())

            x += "cwsq 0.75 raw {} -r {},{},8,{}".format(raw_file, self.img.shape[1], self.img.shape[0], self.ppi)
            o = raw_file + '.wsq'

        # Add other options later if needed.
        os.system(x)
        self.converted = o
        # Now segment
        self.segment()