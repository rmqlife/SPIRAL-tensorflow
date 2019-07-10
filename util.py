import os
import cv2
import numpy as np
import datetime
def timestamp():
    return datetime.datetime.now().strftime("%m%d-%H%M%S")      
    
class Timer:
    def __init__(self):
        self.tic()
        
    def tic(self):
        self._start = self.now()

    def toc(self):
        return str(self.now() - self._start)+'second elapsed'
    
    def now(self):
        import time
        return time.time()


class Recorder:
    def __init__(self):
        self._history = dict()
        self._history['action'] = list()
        self._history['obs'] = list()
        self._history['reward'] = list()
        self._history['done'] = list()
        self._history['info'] = list()
                
    def add(self, action, obs, reward, done, info):
        self._history['action'].append(action)
        self._history['obs'].append(obs)
        self._history['reward'].append(reward)
        self._history['done'].append(done)
        self._history['info'].append(info)

    def save(self, filename):
        if filename.endswith('.gif'):
            import imageio
            with imageio.get_writer(filename, mode='I') as writer:
                for obs in self._history['obs']:
                    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
                    writer.append_data(obs)
        else:
            im = self._history['obs'][-1]
            cv2.imwrite(filename, im)
        # print('Environment Recorder saved to', filename)
        pass
    
    def total_reward(self):
        tt = 0
        for reward in self._history['reward']:
            tt += reward
        return tt
    
    def total_steps(self):
        return len(self._history['action'])


def convert_color_scaler(scaler, type=cv2.COLOR_HSV2BGR):
    scaler= np.uint8([[scaler]]) 
    scaler = cv2.cvtColor(scaler, cv2.COLOR_HSV2BGR)
    return scaler.ravel().tolist()

def im_dist(a, b):
    assert a.shape==b.shape    
    return np.linalg.norm((a.astype('float') - b.astype('float')))

def sample_patch_bw(im, ws, sw=2):
    # stroke width
    for _ in range(1000):
        c = [np.random.randint(im.shape[0]-sw), np.random.randint(im.shape[1])-sw]
        c = [c[0]+sw//2, c[1]+sw//2]
        c = np.array(c)
        val = np.mean(im[c[0]-sw//2:c[0]+sw//2, c[1]-sw//2:c[1]+sw//2])
        if val<50:
            patch = get_patch(im=im, pos=c, ws=ws)
            return patch
        
def sample_patch(im, ws):       
    assert(ws>0)
    c = [np.random.randint(im.shape[0]-2*ws), np.random.randint(im.shape[1]-2*ws)]
    c = [c[0]+ws, c[1]+ws]
    c = np.array(c)
    patch = get_patch(im=im, pos=c, ws=ws)
    return patch

def rotate_flip(im):
    assert im.shape[0]==im.shape[1]
    ims = []
    ws = im.shape[0]//2
    # flip
    for j in range(2):
        if j>0:
            im = cv2.flip(im, j)
        # rotate
        for i in range(4):
            M = cv2.getRotationMatrix2D((ws,ws),90,1)
            im = cv2.warpAffine(im,M,(ws*2+1,ws*2+1))
            ims.append(im)
    return ims

def erode(im, ws=2, iter=1):
    kernel = np.ones((ws,ws),np.uint8)
    im = cv2.erode(im, kernel, iterations=iter);
    return im

def boundingbox(p1,p2):
    xmin = min(p1[0], p2[0])
    xmax = max(p1[0], p2[0])
    ymin = min(p1[1], p2[1])
    ymax = max(p1[1], p2[1])
    return [xmin, xmax, ymin, ymax]


def boundingbox_canvas(p1,p2,r,shape):
    xmin, xmax, ymin, ymax = boundingbox(p1,p2)
    xmin = max(xmin-r, 0)
    ymin = max(ymin-r, 0)
    xmax = min(xmax+r, shape[0]-1)
    ymax = min(ymax+r, shape[1]-1)
    return [xmin, xmax, ymin, ymax]

def mat_center_pos(mat):
    return [mat.shape[0]//2, mat.shape[1]//2]

def mat_random_pos(mat):
    return [np.random.randint(mat.shape[0]), np.random.randint(mat.shape[1])]


def blank(shape, val):
    return (val*np.ones(shape)).astype('uint8')


def test():
    p = blank((50,50,1),val=255)
    from util import show
    print(p.shape)
    show(p)

    print(boundingbox((3,3),(1,1)))
    print(boundingbox((3,3),(5,5)))
    print(boundingbox((3,3),(5,-1)))
    print(boundingbox_canvas(p1=(1,1),p2=(1,1),r=1,shape=[5,5]))

    print(mat_center_pos(np.ones((5,5,5))))
    print(mat_random_pos(np.ones((5,5,5))))


def rgb2rgbgray(im):
    tmp = (255*np.ones(im.shape)).astype('uint8')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    for c in range(3):
        tmp[:,:,c] = im
    return tmp

def mean_color(im):
    avg_color_per_row = np.average(im, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0).astype('uint')
    avg_color = avg_color.tolist()
    return avg_color

def angle_to_coord(a,l):
    aa = 2*np.pi*(a  - 0.5)
    return [l*np.cos(aa), l*np.sin(aa)]


def show(im, colorbar=False):
    import matplotlib.pyplot as plt
    if len(im.shape)==3:
        im2 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im2)
    else:
        plt.imshow(im); 
    if colorbar:
        plt.colorbar()
    plt.show()
    
def show_list(ims):
    import matplotlib.pyplot as plt

    n = len(ims)
    for i in range(n):
        plt.subplot(1, n, 1+i)
        plt.imshow(ims[i])
#         plt.colorbar()
    plt.show()
    
def stack_vector(mat,vec):
    mat = np.vstack((mat,vec)) if mat.size else vec
    return mat
    
def create_folder(name):
    import os, shutil
    if os.path.exists(name):
        shutil.rmtree(name)
    os.mkdir(name)    
    
    
def get_images(home_dir, ext=['png','jpg'], begin=[], number_name=False):
    filelist = get_filelist(home_dir, ext, begin, number_name)
    # print('get images', filelist)
    ims = []
    for fn in filelist:
        im = cv2.imread(fn)
        ims.append(im)
    return ims

def get_filelist(home_dir, ext=['png','jpg'], begin=[], number_name=False):
    filelist = []
    for dirpath, dirnames, filenames in os.walk(home_dir):
        for filename in filenames:
            if filename[0] != '.':
                ext_fag = False
                if len(ext)>0:
                    for e in ext:
                        if filename.endswith(e):
                            ext_fag=True
                            break
                else:
                    ext_fag = True

                begin_fag = False
                if len(begin)>0:
                    for b in begin:
                        if filename.startswith(b):
                            begin_fag=True
                            break
                else:
                    begin_fag = True
                if begin_fag and ext_fag:
                    fn = os.path.join(dirpath,filename)
                    filelist.append(fn)
    filelist = sorted(filelist)
    if number_name:
        filelist = sorted(filelist,key=lambda x: int(os.path.basename(x)[:-4]))
    return filelist


def find_start(im):
    sw = 2
    for _ in range(10000):
        c = [np.random.randint(im.shape[0]-sw), np.random.randint(im.shape[1])-sw]
        c = [c[0]+sw//2, c[1]+sw//2]
        c = np.array(c)
        val = np.mean(im[c[0]-sw//2:c[0]+sw//2, c[1]-sw//2:c[1]+sw//2])
        if np.max(val)<50:
            return c
        
def sample_img(filename):
    im = cv2.imread(filename)
    plt.imshow(im); plt.show()
    patch = get_patch(im=im, pos=find_start(im), ws=5)
    plt.imshow(patch)


def get_patch(im, pos, ws, need_expand=False):
    assert pos[0]<im.shape[0] and pos[0]>=0
    assert pos[1]<im.shape[1] and pos[1]>=0
    if np.isscalar(ws):
        ws = [ws, ws]
    # comparing boundaries
    for i in range(2):
        if int(pos[i]-ws[i])<0 or pos[i] + ws[i]>= im.shape[i]:
            need_expand = True
    
    if need_expand:
        temp_im = 255*np.ones((im.shape[0]+ws[0]*2, im.shape[1]+ws[1]*2, 3)).astype('uint8')
        temp_im[ws[0]:im.shape[0]+ws[0], ws[1]:im.shape[1]+ws[1]] = im
        ret = temp_im[pos[0]:pos[0]+2*ws[0]+1, pos[1]:pos[1]+2*ws[1]+1]
    else:
        ret = im[pos[0]-ws[0]:pos[0]+ws[0]+1, pos[1]-ws[1]:pos[1]+ws[1]+1] 
    return ret


def invert_color(im):
    im = (255 * np.ones(im.shape)).astype('uint8') - im
    return im


def patch_score(im):
    # center of the image
    c = (np.array(im.shape) / 2).astype('uint')
    pts = np.array(np.where(im > 0)).T

    upper_dist = np.linalg.norm(c)

    score = 0
    for pt in pts:
        dist = np.linalg.norm(pt - c)
        s = (upper_dist - dist) / upper_dist
        score += s ** 2
    return score


def find_center(im):
    pts = np.array(np.where(im > 0)).T
    return (np.mean(pts, axis=0)).astype('uint')


def get_salience(im, ws=[100, 100], down_times=4, white_canvas=True):
    if (white_canvas):
        im = invert_color(im)
    for _ in range(down_times):
        im = cv2.pyrDown(im)
        ws = (np.array(ws) / 2).astype('uint')
    max_score = 0
    patch = None
    for _ in range(200):
        pos = sample_pos(im, ws=ws)
        patch = get_patch(im=im, pos=pos, ws=ws)
        score = patch_score(patch)
        if score > max_score:
            max_score = score
            max_pos = pos
    patch = get_patch(im=im, pos=max_pos, ws=ws)
    c = find_center(patch)
    max_pos = (c - ws / 2) + max_pos
    for _ in range(down_times):
        max_pos = max_pos * 2
    return max_pos, score


def fill_patch(im, pos, ws, val):
    im[int(pos[0] - ws[0] / 2):int(pos[0] + ws[0] / 2), int(pos[1] - ws[1] / 2):int(pos[1] + ws[1] / 2)] = val
    return im


def replay_large():
    actions = env.actions
    scale = 4
    im = (255 * np.ones(np.array(env.canvas.shape) * scale)).astype('uint8')
    env = Painting2(im, white_canvas=True, stroke_width=8)
    for action in actions:
        action[1] = int(action[1] * scale)
        action[0] = int(action[0] * scale)
        fn = 'replay.gif'
        env.step(action)
    env.save(os.path.join(folder, fn))


def vec2img(vec):
    vec = (vec * 255).astype('uint8')
    l = np.sqrt(len(vec)).astype('uint')
    return np.reshape(vec, (l, l))


def img2vec(img):
    vec = img.ravel().astype('float') / 255.0
    return vec


def im2bin(im, thresh):
    ret = 255*(im>thresh).astype('uint8')
    return ret

def sample_pos(im, ws=[100,100]):
    ims = im.shape[:2]
    ws = np.array(ws).astype('uint')
    pos = (np.random.rand(2)*(ims-ws)).astype('uint')
    pos = (ws/2+pos).astype('uint')
    # edge
    return pos


def add_border(im, ratio=0.2):
    sp = int(ratio*np.min(im.shape[:2]))
    img = blank((im.shape[0]+sp*2, im.shape[1]+sp*2, im.shape[2]), val=0)
    img[sp:sp+im.shape[0], sp:sp+im.shape[1]]=im
    return img

