#!/usr/bin/env python3
#
# Visualize manifold
#
# Install: pip install numpy scipy scikit-image networkx vtk opencv-python pillow
# Run: python graph_twistor_3d.py --output twistor.png --width 1800
#     python graph_twistor_3d.py --azimuth 25 --elevation 10 --output side.png
#
# mikael.mieskolainen@cern.ch (2026)

import argparse, json
from pathlib import Path
import numpy as np
import networkx as nx
import cv2
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from scipy.interpolate import splprep, splev, RectBivariateSpline, CubicSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from PIL import Image

W,H,UNIT=1408,1056,230.
CONTOURS=[[[1322, 201], [1298, 164], [1268, 127], [1229, 90], [1187, 60], [1162, 46], [1121, 29], [1078, 18], [1040, 14], [993, 16], [944, 26], [896, 43], [860, 60], [822, 81], [783, 107], [730, 150], [669, 209], [661, 215], [639, 224], [624, 223], [605, 217], [549, 192], [489, 170], [456, 161], [422, 155], [393, 152], [349, 153], [312, 159], [268, 172], [229, 189], [209, 200], [181, 218], [145, 247], [116, 277], [93, 306], [70, 342], [53, 375], [42, 402], [29, 445], [23, 474], [19, 514], [20, 554], [25, 593], [35, 636], [49, 678], [69, 725], [92, 768], [119, 809], [152, 850], [187, 885], [225, 915], [266, 938], [298, 949], [316, 952], [348, 952], [365, 949], [394, 939], [427, 921], [456, 900], [496, 866], [592, 770], [623, 745], [651, 727], [657, 727], [669, 733], [702, 757], [778, 833], [833, 880], [873, 907], [909, 926], [945, 939], [966, 944], [990, 947], [1033, 945], [1084, 932], [1127, 913], [1174, 884], [1219, 847], [1259, 805], [1302, 750], [1328, 708], [1350, 664], [1370, 610], [1381, 566], [1388, 520], [1390, 496], [1390, 441], [1388, 416], [1377, 348], [1364, 299], [1345, 247]], [[563, 352], [562, 361], [505, 465], [482, 496], [456, 523], [420, 550], [390, 564], [373, 568], [351, 568], [339, 565], [331, 560], [323, 529], [323, 504], [326, 487], [335, 463], [344, 447], [361, 425], [377, 409], [396, 394], [424, 377], [451, 365], [472, 358], [516, 350], [552, 350]], [[1061, 342], [1063, 347], [1062, 364], [1059, 375], [1050, 394], [1034, 416], [1022, 428], [1001, 444], [975, 458], [952, 467], [868, 493], [792, 522], [784, 521], [784, 515], [791, 486], [807, 437], [837, 429], [863, 414], [888, 392], [919, 356], [937, 338], [954, 325], [965, 319], [981, 314], [1008, 313], [1029, 318], [1049, 329]]]

def unit(a):
    a=np.asarray(a,float); return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-14)

def xy_pixel(x,y):return np.array([(x-W/2)/UNIT,(H/2-y)/UNIT])

def loop(p):
    p=np.asarray(p,float);p=p[:-1] if np.allclose(p[0],p[-1]) else p;tck,_=splprep(np.vstack([p,p[0]]).T,s=.7,per=True)
    return np.array(splev(np.linspace(0,1,1600,endpoint=False),tck)).T

class Component:
    """F(x,y,z)=(z-m(x,y))²-q(x,y)=0, with q>0 inside the domain."""
    def __init__(self,res=420,contours=CONTOURS,depth=.88,kind="ring"):
        self.x=np.linspace(-W/2/UNIT-.12,W/2/UNIT+.12,res);hx=self.x[1]-self.x[0]
        self.y=np.linspace(-H/2/UNIT-.12,H/2/UNIT+.12,round((H/UNIT+.24)/hx)+1);hy=self.y[1]-self.y[0]
        X,Y=np.meshgrid(self.x,self.y);xy_grid=np.c_[X.ravel(),Y.ravel()]
        # Smooth sub-pixel signed distance from densely sampled spline rims.
        curves=[];inward=[];splines=[];orient=[]
        for j,c in enumerate(contours):
            xy=np.array([xy_pixel(*p) for p in loop(c)])
            tang=unit(np.roll(xy,-1,axis=0)-np.roll(xy,1,axis=0))
            area=np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))
            normal=np.c_[-tang[:,1],tang[:,0]]*np.sign(area)*(1 if j==0 else -1)
            curves.append(xy);inward.append(normal)
            tck,_=splprep(np.vstack([xy,xy[0]]).T,u=np.linspace(0,1,len(xy)+1),s=0,per=True);splines.append(tck);orient.append(np.sign(area)*(1 if j==0 else -1))
        rim=np.concatenate(curves);rn=np.concatenate(inward)
        dist,near=cKDTree(rim).query(xy_grid);sdflat=dist*np.sign(np.einsum('ij,ij->i',xy_grid-rim[near],rn[near]))
        # Refine each nearest-point query to the continuous spline curve. The
        # sampled contour alone creates periodic small normal-map artifacts
        # when viewed from the side. Newton's method removes this quantization.
        nseg=len(curves[0]);tref=(near%nseg)/nseg;rim_id=near//nseg
        for j,tck in enumerate(splines):
            use=(rim_id==j)&(dist<.24);t=tref[use].copy();points=xy_grid[use]
            for _ in range(5):
                c=np.array(splev(t,tck)).T;dc=np.array(splev(t,tck,der=1)).T;ddc=np.array(splev(t,tck,der=2)).T
                num=np.einsum('ij,ij->i',c-points,dc);den=np.einsum('ij,ij->i',dc,dc)+np.einsum('ij,ij->i',c-points,ddc)
                t=(t-np.clip(num/np.maximum(den,1e-9),-.01,.01))%1.
            c=np.array(splev(t,tck)).T;dc=unit(np.array(splev(t,tck,der=1)).T);nn=np.c_[-dc[:,1],dc[:,0]]*orient[j]
            sdflat[use]=np.einsum('ij,ij->i',points-c,nn);tref[use]=t
        sd=sdflat.reshape(X.shape)
        inside=sd>0;idx=np.full(inside.shape,-1,int);idx[inside]=np.arange(inside.sum())
        iy,ix=np.where(inside);ii=idx[inside];rows=[];cols=[];vals=[];diag=np.zeros(len(ii))
        # Shortley-Weller finite differences with subcell Dirichlet boundaries.
        for (dy,dx,h) in [(0,1,hx),(1,0,hy)]:
            im=idx[iy-dy,ix-dx];ip=idx[iy+dy,ix+dx];d=sd[iy,ix]
            hm=h*np.where(im>=0,1,np.clip(d/np.maximum(d-sd[iy-dy,ix-dx],1e-15),.04,1))
            hp=h*np.where(ip>=0,1,np.clip(d/np.maximum(d-sd[iy+dy,ix+dx],1e-15),.04,1))
            diag+=2/(hm*hp)
            for jj,hh in [(im,hm),(ip,hp)]:
                ok=jj>=0;rows.append(ii[ok]);cols.append(jj[ok]);vals.append((-2/(hh*(hm+hp)))[ok])
        rows.append(ii);cols.append(ii);vals.append(diag)
        A=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(len(ii),len(ii))).tocsr()
        q=np.zeros(inside.shape);q[inside]=spsolve(A,np.ones(len(ii)))
        # Remove discretization corrugation along the rim using a smooth
        # quadratic normal-coordinate extension, exactly zero at d=0.
        q0=RectBivariateSpline(self.x,self.y,q.T);d1,d2=.052,.104
        p1=rim+d1*rn;p2=rim+d2*rn
        q1=q0.ev(p1[:,0],p1[:,1]);q2=q0.ev(p2[:,0],p2[:,1]);b=(q2/d2-q1/d1)/(d2-d1);a=q1/d1-b*d1
        off=0
        for c in curves:
            sl=slice(off,off+len(c));a[sl]=gaussian_filter1d(a[sl],12,mode='wrap');b[sl]=gaussian_filter1d(b[sl],12,mode='wrap');off+=len(c)
        a=np.maximum(a,.025);av=a[near].copy();bv=b[near].copy()
        # Continuous coefficients along the same rim parameter.
        for j in range(len(curves)):
            use=(rim_id==j)&(dist<.24);sl=slice(j*nseg,(j+1)*nseg);aa=a[sl];bb=b[sl]
            ts=np.linspace(0,1,nseg+1);av[use]=CubicSpline(ts,np.r_[aa,aa[0]],bc_type='periodic')(tref[use]);bv[use]=CubicSpline(ts,np.r_[bb,bb[0]],bc_type='periodic')(tref[use])
        av=av.reshape(X.shape);bv=bv.reshape(X.shape)
        qe=av*sd+bv*np.clip(sd,-.04,.14)*sd
        w=np.clip((sd-.045)/.095,0,1);w=w*w*(3-2*w)
        q=(1-w)*qe+w*q;q*=depth**2/q.max()
        X,Y=np.meshgrid(self.x,self.y)
        if kind=="ring":
            mid=.17*np.sin(1.05*X+.62*Y)+.11*np.sin(1.8*Y-.45*X)
            mid+=.20*np.exp(-.5*((Y+.43-.30*X)/.32)**2-.5*(X/1.42)**2)
            mid+=.16*np.exp(-.5*((Y-.65+.12*X)/.33)**2-.5*(X/1.2)**2)
        else:
            mid=-.82+.15*np.sin(1.2*X+.4*Y)+.12*np.tanh((Y-.1)*1.4)+.38*np.exp(-.5*((X+1.05)/.75)**2-.5*((Y+.76)/.37)**2)
        self.q=RectBivariateSpline(self.x,self.y,q.T);self.m=RectBivariateSpline(self.x,self.y,mid.T)
        self.qg=q;self.mg=mid
    def fg(self,p):
        x,y,z=np.moveaxis(np.asarray(p,float),-1,0);m=self.m.ev(x,y);d=z-m
        f=d*d-self.q.ev(x,y)
        g=np.stack([-2*d*self.m.ev(x,y,dx=1)-self.q.ev(x,y,dx=1),-2*d*self.m.ev(x,y,dy=1)-self.q.ev(x,y,dy=1),2*d],axis=-1)
        return f,g
    def normal(self,p):return unit(self.fg(p)[1])
    def front(self,x,y):
        q=self.q.ev(x,y)
        return None if q<=0 else np.array([x,y,self.m.ev(x,y)+np.sqrt(q)])
    def chart(self,c,u,v,xy):
        # Lift the planar chart along its centre normal, using the exact 3D F.
        n=self.normal(c);p=c+xy[...,0,None]*u+xy[...,1,None]*v
        for _ in range(16):
            f,g=self.fg(p);d=g@n
            if np.any(d<=1e-7):return None
            step=f/d
            if np.max(np.abs(step))>.8:return None
            p-=step[...,None]*n
            if np.max(np.abs(step))<2e-9:break
        f,g=self.fg(p)
        if np.max(np.abs(f))>1e-7 or np.min(unit(g)@n)<.38:return None
        return p

# The outer loop encloses one aperture; the recessed bridge divides it in two.
APERTURE = [[563, 352], [620, 365], [690, 395], [754, 428], [802, 435], [858, 407], [909, 350], [955, 322], [995, 313], [1044, 329], [1063, 350], [1052, 387], [1029, 420], [994, 449], [936, 471], [870, 492], [800, 519], [735, 547], [665, 574], [596, 599], [527, 623], [460, 642], [407, 645], [365, 628], [338, 597], [324, 555], [322, 514], [336, 468], [365, 428], [405, 393], [451, 371], [563, 352]]
BRIDGE = [[665,258],[610,286],[578,327],[563,352],[562,361],[535,413],
          [505,465],[482,496],[456,523],[420,550],[390,564],[373,568],
          [351,568],[339,565],[326,552],[299,574],[310,639],[346,683],[402,723],[472,736],[552,708],
          [631,671],[705,609],[750,568],[784,521],[791,486],[807,437],
          [837,429],[863,414],[888,392],[905,359],[900,307],[857,270],
          [799,241],[720,239]]

class Surface:
    """Smooth union of two closed implicit volumes; real depth and occlusion."""
    def __init__(self,res=420):
        self.loop=Component(res,[CONTOURS[0], APERTURE],1.02,"ring")
        self.bridge=Component(res,[BRIDGE],.71,"bridge")
        self.x,self.y=self.loop.x,self.loop.y
        self.blend=.15
    def combine(self,a,b):
        k=self.blend; h=np.maximum(k-np.abs(a-b),0)/k
        return np.minimum(a,b)-(k/6)*h**3
    def fg(self,p):
        a,ga=self.loop.fg(p);b,gb=self.bridge.fg(p)
        h=np.maximum(1-np.abs(a-b)/self.blend,0)
        w=np.where(a<=b,1-.5*h*h,.5*h*h)
        return self.combine(a,b),w[...,None]*ga+(1-w[...,None])*gb
    def normal(self,p):return unit(self.fg(p)[1])
    def front(self,x,y):
        roots=[]
        for c in (self.loop,self.bridge):
            q=c.q.ev(x,y)
            if q>0:roots.append(c.m.ev(x,y)+np.sqrt(q))
        if not roots:return None
        p=np.array([x,y,max(roots)])
        for _ in range(12):
            f,g=self.fg(p)
            if abs(f)<1e-10:return p
            if g[2]<1e-8:return None
            p[2]-=f/g[2]
        return p if abs(self.fg(p)[0])<1e-7 else None
    chart=Component.chart
    def mesh(self):
        z=np.linspace(-2.05,1.70,211,dtype=np.float32)
        def vol(c):return (z[None,None,:]-c.mg.T[:,:,None].astype(np.float32))**2-c.qg.T[:,:,None].astype(np.float32)
        a=vol(self.loop);b=vol(self.bridge)
        h=np.abs(a-b);h*=(-1/self.blend);h+=1;np.maximum(h,0,out=h);h=h**3;h*=self.blend/6
        np.minimum(a,b,out=a);a-=h;del b,h
        v,f,_,_=marching_cubes(a,0,spacing=(self.x[1]-self.x[0],self.y[1]-self.y[0],float(z[1]-z[0])),allow_degenerate=False);del a
        v=v.astype(float)+np.array([self.x[0],self.y[0],z[0]])
        for _ in range(4):
            ff,g=self.fg(v);v-=ff[:,None]*g/np.maximum((g*g).sum(1,keepdims=True),1e-12)
        return v,f.astype(np.int64),self.normal(v)

def ring(n,r=1,phase=np.pi/2):
    a=phase+np.arange(n)*2*np.pi/n;return r*np.c_[np.cos(a),np.sin(a)]

def tutte_layout(g):
    planar,emb=nx.check_planarity(g)
    if not planar:raise ValueError('Tutte embedding requires planarity')
    seen=set();faces=[]
    for u,v in emb.edges():
        if (u,v) not in seen:faces.append(emb.traverse_face(u,v,seen))
    outer=max(faces,key=len);boundary=dict(zip(outer,ring(len(outer))))
    interior=[u for u in g if u not in boundary];ix={u:i for i,u in enumerate(interior)}
    A=np.zeros((len(ix),len(ix)));rhs=np.zeros((len(ix),2))
    for u,i in ix.items():
        A[i,i]=g.degree(u)
        for v in g[u]:
            if v in ix:A[i,ix[v]]-=1
            else:rhs[i]+=boundary[v]
    return {**boundary,**dict(zip(interior,np.linalg.solve(A,rhs)))}

def template(kind):
    if kind.startswith('cycle'):
        n=int(kind[5:]);g=nx.cycle_graph(n);pos=dict(enumerate(ring(n)))
    elif kind.startswith('complete'):
        n=int(kind[8:]);g=nx.complete_graph(n);p=np.vstack([ring(3),[[0,0]]]) if n==4 else ring(n)
        if n>=6:
            a=np.pi/2+np.arange(n)*2*np.pi/n+.10*np.sin(np.arange(n)*2.17+.3);p=np.c_[np.cos(a),np.sin(a)]
        pos=dict(enumerate(p))
    elif kind.startswith('wheel'):
        n=int(kind[5:]);g=nx.wheel_graph(n);pos=dict(enumerate(np.vstack([[0,0],ring(n-1)])))
    elif kind.startswith('star'):
        n=int(kind[4:]);g=nx.star_graph(n);pos=dict(enumerate(np.vstack([[0,0],ring(n)])))
    elif kind=='petersen':
        g=nx.petersen_graph();pos=dict(enumerate(np.vstack([ring(5),ring(5,.43)])))
    elif kind in ('cube','prism'):
        n=4 if kind=='cube' else 5;g=nx.Graph();g.add_edges_from([(i,(i+1)%n) for i in range(n)]+[(n+i,n+(i+1)%n) for i in range(n)]+[(i,i+n) for i in range(n)])
        pos=dict(enumerate(np.vstack([ring(n,1,np.pi/4),ring(n,.46,np.pi/4)])))
    elif kind.startswith('grid'):
        a,b=map(int,kind[4:].split('x'));g=nx.grid_2d_graph(a,b);pos={i:i for i in g}
    elif kind.startswith('ladder'):
        n=int(kind[6:]);g=nx.ladder_graph(n);pos={i:(i,.5) for i in range(n)};pos.update({i+n:(i,-.5) for i in range(n)})
    elif kind=='tree':
        g=nx.balanced_tree(2,3);pos={}
        for i in g:
            l=int(np.floor(np.log2(i+1)));j=i-(2**l-1);pos[i]=((j+.5)/2**l*2-1,1-.52*l)
    elif kind=='branch':
        g=nx.Graph([(0,1),(1,2),(2,3),(3,4),(2,5),(4,6),(4,7)]);pos=dict(enumerate([[-1,0],[-.65,.15],[-.28,.05],[.12,-.12],[.55,.05],[-.20,.66],[.88,.50],[.93,-.42]]))
    elif kind=='house':
        g=nx.Graph([(0,1),(1,2),(2,3),(3,0),(2,4),(3,4)]);pos=dict(enumerate([[-1,-.7],[1,-.7],[1,.6],[-1,.6],[0,1.5]]))
    elif kind=='bipartite':
        g=nx.complete_bipartite_graph(3,4);pos=dict(enumerate([[-.8,-1],[-.8,.06],[-.8,1],[.8,-1.04],[.8,-.31],[.8,.36],[.8,1.02]]))
    elif kind=='hex':
        g=nx.hexagonal_lattice_graph(2,2);pos=nx.get_node_attributes(g,'pos')
    elif kind=='triangular':
        g=nx.Graph();pos={(r,j):(j-r/2,-r*np.sqrt(3)/2) for r in range(5) for j in range(r+1)};g.add_nodes_from(pos)
        for r,j in pos:
            for t in [(r,j+1),(r+1,j),(r+1,j+1)]:
                if t in pos:g.add_edge((r,j),t)
    elif kind=='path':
        g=nx.path_graph(6);pos={i:(i, .22*np.sin(i*.9)) for i in g}
    elif kind=='diamond':
        g=nx.Graph([(0,1),(1,2),(2,3),(3,0),(0,2)]);pos=dict(enumerate([[0,1],[.85,0],[0,-1],[-.85,0]]))
    elif kind=='bowtie':
        g=nx.Graph([(0,1),(1,2),(2,0),(0,3),(3,4),(4,0)]);pos=dict(enumerate([[0,0],[-1,-.68],[-1,.68],[1,.68],[1,-.68]]))
    elif kind=='bull':
        g=nx.bull_graph();pos=dict(enumerate([[0,.94],[-.8,0],[.8,0],[-1.27,-.75],[1.27,-.75]]))
    elif kind=='paw':
        g=nx.Graph([(0,1),(1,2),(2,0),(0,3)]);pos=dict(enumerate([[0,.26],[-.82,-.75],[.82,-.75],[0,1.25]]))
    elif kind=='fan':
        g=nx.Graph();g.add_edges_from((0,i) for i in range(1,7));g.add_edges_from((i,i+1) for i in range(1,6));pos={0:(0,-.63)};pos.update({i:(np.cos(a),np.sin(a)) for i,a in enumerate(np.linspace(.16,np.pi-.16,6),1)})
    elif kind=='friendship':
        g=nx.Graph();g.add_edges_from((0,i) for i in range(1,9));g.add_edges_from((i,i+1) for i in range(1,9,2));a=np.arange(4)*np.pi/2;ang=np.ravel(np.c_[a-.32,a+.32]);pos=dict(enumerate(np.vstack([[0,0],np.c_[np.cos(ang),np.sin(ang)]])))
    elif kind=='sunlet':
        n=5;g=nx.cycle_graph(n);g.add_edges_from((i,i+n) for i in range(n));pos=dict(enumerate(np.vstack([ring(n,.59),ring(n)])))
    elif kind=='gear':
        n=5;g=nx.cycle_graph(2*n);g.add_edges_from((2*n,i) for i in range(0,2*n,2));pos=dict(enumerate(np.vstack([ring(2*n)*np.where(np.arange(2*n)%2,.98,.70)[:,None],[[0,0]]])))
    elif kind=='theta':
        g=nx.Graph([(0,1),(1,2),(2,3),(0,4),(4,5),(5,3),(0,6),(6,7),(7,3)]);pos=dict(enumerate([[-1,0],[-.38,.68],[.38,.68],[1,0],[-.38,0],[.38,0],[-.38,-.68],[.38,-.68]]))
    elif kind=='barbell':
        g=nx.Graph([(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,6),(6,4)]);pos=dict(enumerate([[-1,-.60],[-1,.60],[-.52,0],[0,0],[.52,0],[1,.60],[1,-.60]]))
    elif kind=='lollipop':
        g=nx.Graph([(0,1),(1,2),(2,3),(3,0),(2,4),(4,5),(5,6)]);pos=dict(enumerate([[-.85,.65],[-.85,-.65],[-.20,-.65],[-.20,.65],[.35,-.36],[.75,.04],[1.05,.48]]))
    elif kind=='windmill':
        g=nx.Graph();pos={0:(0,0)}
        for k in range(3):
            a=2*np.pi*k/3;ids=[0,3*k+1,3*k+2,3*k+3]
            g.add_edges_from((u,v) for j,u in enumerate(ids) for v in ids[j+1:]);pos.update({ids[j+1]:(r*np.cos(a+t),r*np.sin(a+t)) for j,(r,t) in enumerate([(1,-.36),(.67,0),(1,.36)])})
    elif kind=='octahedral':
        g=nx.Graph();g.add_edges_from([(i,(i+1)%3) for i in range(3)]+[(i+3,(i+1)%3+3) for i in range(3)]+[(i,i+3) for i in range(3)]+[(i,(i-1)%3+3) for i in range(3)])
        pos=dict(enumerate(np.vstack([ring(3),ring(3,.35,np.pi/2+np.pi/3)])))
    elif kind=='antiprism':
        n=4;g=nx.Graph();g.add_edges_from([(i,(i+1)%n) for i in range(n)]+[(i+n,(i+1)%n+n) for i in range(n)]+[(i,i+n) for i in range(n)]+[(i,(i-1)%n+n) for i in range(n)])
        pos=dict(enumerate(np.vstack([ring(n),ring(n,.47,np.pi/2+np.pi/n)])))
    elif kind in ('dodecahedral','icosahedral','truncated_tetrahedron','truncated_cube'):
        generators={'dodecahedral':nx.dodecahedral_graph,'icosahedral':nx.icosahedral_graph,'truncated_tetrahedron':nx.truncated_tetrahedron_graph,'truncated_cube':nx.truncated_cube_graph}
        g=generators[kind]();pos=tutte_layout(g)
    elif kind=='bipartite33':
        g=nx.complete_bipartite_graph(3,3);pos=dict(enumerate([[-.8,-1],[-.8,.03],[-.8,1],[.8,-.95],[.8,-.08],[.8,1.04]]))
    elif kind=='crown':
        g=nx.Graph([(i,4+j) for i in range(4) for j in range(4) if i!=j]);pos=dict(enumerate([[-.75,-1],[-.75,-.31],[-.75,.34],[-.75,1],[.75,-.94],[.75,-.28],[.75,.28],[.75,1.04]]))
    elif kind=='circulant':
        n=7;g=nx.circulant_graph(n,[1,2]);pos=dict(enumerate(ring(n)))
    elif kind=='mobius':
        n=8;g=nx.cycle_graph(n);g.add_edges_from((i,i+n//2) for i in range(n//2));a=np.pi/2+np.arange(n)*2*np.pi/n+.09*np.sin(np.arange(n)*2.19);pos=dict(enumerate(np.c_[np.cos(a),np.sin(a)]))
    elif kind=='ternary':
        g=nx.balanced_tree(3,2);pos={0:(0,1)}
        for i in range(3):
            pos[i+1]=((i-1)*.68,.35)
            for j in range(3):pos[4+3*i+j]=((i-1)*.78+(j-1)*.24,-.65)
    elif kind=='rhombille':
        # A finite rhombic tiling graph in exact oblique coordinates.
        g=nx.grid_2d_graph(3,4);pos={(i,j):(i+.5*j,.8660254038*j) for i,j in g}
    elif kind=='trihex':
        # Hexagonal wheel surrounded by six equilateral outer triangles.
        g=nx.cycle_graph(6);g.add_edges_from((6,i) for i in range(6));g.add_edges_from((7+i,j) for i in range(6) for j in (i,(i+1)%6));pos=dict(enumerate(np.vstack([ring(6,.59),[[0,0]],ring(6,1,np.pi/2+np.pi/6)])))
    else:raise ValueError(kind)
    ids=list(g.nodes());p=np.array([pos[i] for i in ids],float);p-=p.mean(0);p/=np.linalg.norm(p,axis=1).max();ix={k:i for i,k in enumerate(ids)}
    e=np.array([(ix[a],ix[b]) for a,b in g.edges()],int)
    # Audit simple graph and nonincident vertex-edge clearance in the chart.
    assert len({tuple(sorted(x)) for x in e})==len(e) and not np.any(e[:,0]==e[:,1])
    for i,j in e:
        a,b=p[i],p[j];t=np.clip(((p-a)@(b-a))/np.dot(b-a,b-a),0,1);d=np.linalg.norm(p-a-t[:,None]*(b-a),axis=1);d[[i,j]]=np.inf
        if d.min()<.025:raise ValueError('Degenerate drawing: '+kind)
    assert nx.is_connected(g)
    if nx.check_planarity(g)[0]:
        for k,(i,j) in enumerate(e):
            a,b=p[i],p[j];d=b-a
            for l,m in e[:k]:
                if len({i,j,l,m})<4:continue
                c=p[l];h=p[m]-c;det=d[0]*h[1]-d[1]*h[0]
                if abs(det)<1e-12:continue
                ac=c-a;t=(ac[0]*h[1]-ac[1]*h[0])/det;u=(ac[0]*d[1]-ac[1]*d[0])/det
                if 1e-9<t<1-1e-9 and 1e-9<u<1-1e-9:
                    # Intentionally nonplanar-looking drawings must not occur
                    # for planar templates; choose a different embedding.
                    raise ValueError('Crossing in planar drawing: '+kind)
    return p,e

PALETTE={'purple':(.39,.045,.68),'blue':(.04,.25,.73),'navy':(.055,.09,.59),'green':(.025,.46,.055),'magenta':(.64,.025,.39),'orange':(.84,.27,.015),'violet':(.34,.045,.57)}
# Centres guide placement in the reference view; all actual edges live in 3D.
SPECS=[('cycle10', 222, 273, 66, 0, 'orange'), ('complete5', 180, 433, 72, 0.04, 'purple'), ('wheel11', 144, 589, 65, 0, 'navy'), ('grid4x4', 258, 752, 94, 0.05, 'purple'), ('star7', 276, 546, 45, 0.06, 'magenta'), ('prism', 344, 368, 53, 0.2, 'blue'), ('bull', 122, 718, 40, -0.5, 'green'), ('friendship', 255, 869, 43, -0.3, 'green'), ('diamond', 368, 857, 40, 0.05, 'orange'), ('hex', 460, 742, 82, -0.05, 'blue'), ('sunlet', 351, 197, 67, -0.13, 'purple'), ('paw', 493, 217, 41, 0, 'green'), ('windmill', 454, 318, 53, -0.36, 'magenta'), ('cycle6', 577, 312, 39, 0.18, 'navy'), ('tree', 751, 307, 101, 0.06, 'green'), ('ladder4', 697, 229, 41, 0.65, 'orange'), ('cycle5', 840, 173, 48, 0, 'purple'), ('hex', 933, 89, 75, -0.15, 'violet'), ('bull', 1113, 115, 49, -0.12, 'orange'), ('rhombille', 1111, 49, 45, -0.26, 'blue'), ('complete6', 994, 219, 88, 0.13, 'navy'), ('grid4x5', 1228, 232, 112, 0.04, 'blue'), ('bowtie', 873, 290, 37, -0.05, 'magenta'), ('gear', 1082, 293, 30, -0.1, 'violet'), ('circulant', 1147, 386, 74, 0.02, 'navy'), ('lollipop', 1271, 357, 43, -0.2, 'orange'), ('cycle6', 1350, 431, 31, 0.12, 'green'), ('petersen', 1060, 502, 72, 0, 'blue'), ('cube', 1253, 500, 67, 0.06, 'magenta'), ('cycle8', 1158, 605, 49, 0.1, 'blue'), ('bipartite', 1006, 642, 61, -0.1, 'violet'), ('barbell', 903, 523, 56, 0.35, 'magenta'), ('theta', 840, 613, 58, 0.5, 'magenta'), ('complete4', 1170, 727, 48, 0.06, 'purple'), ('triangular', 1077, 798, 112, 0.05, 'orange'), ('trihex', 892, 758, 87, -0.22, 'green'), ('ladder5', 650, 666, 91, 0.35, 'purple'), ('hex', 533, 574, 71, 0.24, 'green'), ('antiprism', 609, 486, 61, -0.06, 'orange'), ('rhombille', 763, 567, 56, 0.5, 'orange'), ('house', 677, 445, 35, 0, 'green'), ('cycle4', 604, 397, 31, 0.02, 'violet'), ('ladder3', 428, 577, 43, 0.5, 'blue'), ('path', 527, 654, 42, 0.5, 'violet'), ('bowtie', 466, 864, 37, 0.52, 'magenta'), ('diamond', 728, 728, 36, -0.4, 'orange'), ('theta', 784, 807, 46, -0.6, 'blue'), ('octahedral', 878, 882, 43, -0.28, 'violet'), ('dodecahedral', 1014, 914, 58, 0.04, 'purple'), ('rhombille', 1245, 791, 55, 0.68, 'magenta'), ('truncated_tetrahedron', 1308, 620, 65, 0.25, 'purple'), ('path', 1300, 119, 38, -0.5, 'magenta'), ('gear', 92, 360, 40, 0.6, 'violet'), ('complete4', 58, 497, 42, 0.4, 'navy'), ('lollipop', 136, 817, 37, -0.52, 'blue'), ('cycle5', 52, 604, 25, 0.1, 'orange'), ('fan', 341, 923, 44, 0.25, 'blue'), ('diamond', 551, 776, 31, 0.5, 'violet'), ('paw', 1169, 876, 31, 0.28, 'blue'), ('branch', 772, 145, 35, 0.75, 'green')]

def build_graphs(s,seed):
    rng=np.random.default_rng(seed);graphs=[];occupied=[]
    for kind,px,py,rad,ang,col in SPECS:
        p,e=template(kind);R=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]]);p=p@R.T
        accepted=None;offsets=[(0,0)]+[tuple(rng.uniform(-a,a,2)) for a in (10,20,30) for _ in range(5)]
        for fac in (1,.88,.76,.64):
            for dx,dy in offsets:
                c=s.front(*xy_pixel(px+dx,py+dy))
                if c is None:continue
                n=s.normal(c)
                if n[2]<.25:continue
                u=unit(np.array([1.,0,0])-n*n[0]);v=unit(np.cross(n,u));r=rad/UNIT*fac;coords=p*r
                lines=np.array([np.linspace(coords[i],coords[j],81) for i,j in e]);grid=np.array([(i,j) for i in np.linspace(-1,1,7) for j in np.linspace(-1,1,7) if i*i+j*j<1.02])*r
                q=s.chart(c,u,v,np.vstack([coords,lines.reshape(-1,2),grid]))
                if q is None or s.normal(q)[:,2].min()<.10:continue
                nodes=q[:len(p)];paths=q[len(p):len(p)+lines.size//2].reshape(len(e),81,3);screen=q[:len(p)+lines.size//2,:2]
                nr=np.clip((.056 if len(p)>=16 else .077)*r,.010,.025);tr=.30*nr
                # Projected clearance from nonincident vertices and other motifs.
                good=True
                for k,(i,j) in enumerate(e):
                    d=np.linalg.norm(nodes[:,None,:2]-paths[k,None,:,:2],axis=-1).min(1);d[[i,j]]=np.inf
                    if d.min()<1.4*nr:good=False;break
                if not good:continue
                if min(np.linalg.norm(nodes[i,:2]-nodes[j,:2]) for i in range(len(nodes)) for j in range(i))<2.5*nr:continue
                lo=screen.min(0);hi=screen.max(0)
                for l,h,tree,oldr in occupied:
                    pad=nr+oldr+.025
                    if np.any(hi+pad<l) or np.any(h+pad<lo):continue
                    if tree.query(screen,k=1)[0].min()<pad:good=False;break
                if not good:continue
                accepted=(nodes,paths,nr,tr,screen,lo,hi,coords);break
            if accepted is not None:break
        if accepted is None:continue
        nodes,paths,nr,tr,screen,lo,hi,coords=accepted;occupied.append((lo,hi,cKDTree(screen),nr))
        # A nonplanar graph drawing in a disk has unavoidable crossings.
        # Give one edge a small normal-direction overpass, not a spurious node.
        t=np.linspace(0,1,paths.shape[1]);lift=np.full(paths.shape[:2],.82*tr)
        for k,(i,j) in enumerate(e):
            a,b=coords[i],coords[j];d=b-a
            for l,(m,n) in enumerate(e[:k]):
                if len({i,j,m,n})<4:continue
                c=coords[m];h=coords[n]-c;det=d[0]*h[1]-d[1]*h[0]
                if abs(det)<1e-12:continue
                ac=c-a;tk=(ac[0]*h[1]-ac[1]*h[0])/det;tl=(ac[0]*d[1]-ac[1]*d[0])/det
                if .015<tk<.985 and .015<tl<.985:
                    sigma=max(.014,2.1*tr/np.linalg.norm(d))
                    lower=np.interp(tl,t,lift[l]);upper=np.interp(tk,t,lift[k])
                    amp=max(0.,lower+3.2*tr-upper)
                    lift[k]+=amp*np.exp(-.5*((t-tk)/sigma)**2)
        graphs.append(dict(kind=kind,color=PALETTE[col],nodes=nodes+s.normal(nodes)*(.65*nr),paths=paths+s.normal(paths)*lift[...,None],edges=e,surface_nodes=nodes,surface_paths=paths,node_radius=float(nr),tube_radius=float(tr)))
    return graphs

def polydata(v,f,n=None):
    p=vtk.vtkPolyData();pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(np.ascontiguousarray(v),deep=True));p.SetPoints(pts)
    c=vtk.vtkCellArray();c.SetData(numpy_to_vtkIdTypeArray(np.arange(0,3*len(f)+1,3,dtype=np.int64),deep=True),numpy_to_vtkIdTypeArray(np.ascontiguousarray(f,dtype=np.int64).ravel(),deep=True));p.SetPolys(c)
    if n is not None:
        a=numpy_to_vtk(np.ascontiguousarray(n,dtype=np.float32),deep=True);a.SetName('Normals');p.GetPointData().SetNormals(a)
    return p

def actor(p,color,ambient=.23,diffuse=.77,specular=.20,power=35):
    m=vtk.vtkPolyDataMapper();m.SetInputData(p);a=vtk.vtkActor();a.SetMapper(m);pr=a.GetProperty();pr.SetColor(*color);pr.SetInterpolationToPhong();pr.SetAmbient(ambient);pr.SetDiffuse(diffuse);pr.SetSpecular(specular);pr.SetSpecularPower(power);return a

def graph_geometry(g):
    pts=vtk.vtkPoints();lines=vtk.vtkCellArray()
    for path in g['paths']:
        lines.InsertNextCell(len(path))
        for p in path:lines.InsertCellPoint(pts.InsertNextPoint(*p))
    pl=vtk.vtkPolyData();pl.SetPoints(pts);pl.SetLines(lines);tube=vtk.vtkTubeFilter();tube.SetInputData(pl);tube.SetRadius(g['tube_radius']);tube.SetNumberOfSides(12);tube.CappingOn()
    pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(np.ascontiguousarray(g['nodes']),deep=True));p=vtk.vtkPolyData();p.SetPoints(pts)
    sphere=vtk.vtkSphereSource();sphere.SetRadius(g['node_radius']);sphere.SetThetaResolution(20);sphere.SetPhiResolution(16)
    glyph=vtk.vtkGlyph3D();glyph.SetInputData(p);glyph.SetSourceConnection(sphere.GetOutputPort());glyph.ScalingOff()
    app=vtk.vtkAppendPolyData();app.AddInputConnection(tube.GetOutputPort());app.AddInputConnection(glyph.GetOutputPort());app.Update();return app.GetOutput()

def floor(v,f):
    """Soft cast shadow: actual geometry projected onto a 3D stage plane."""
    th=np.deg2rad(13);n=np.array([0,np.cos(th),np.sin(th)]);ex=np.array([1.,0,0]);ez=np.array([0,-np.sin(th),np.cos(th)])
    height=np.min(v@n)-.045;origin=n*height;xmin,xmax,zmin,zmax=-12,12,-22,22;N,M=1000,1400
    X,Z=np.meshgrid(np.linspace(xmin,xmax,N),np.linspace(zmin,zmax,M));Y=origin[1]+Z*ez[1];t=np.clip((-.7-Y)/1.8,0,1)
    base=np.stack([.13+.12*t,.12+.09*t,.165+.135*t],-1)
    dec=vtk.vtkQuadricDecimation();dec.SetInputData(polydata(v,f));dec.SetTargetReduction(.95);dec.Update();pp=dec.GetOutput();vv=vtk_to_numpy(pp.GetPoints().GetData());ff=vtk_to_numpy(pp.GetPolys().GetConnectivityArray()).reshape(-1,3)
    shadow=np.zeros((M,N),np.float32);angles=np.linspace(0,2*np.pi,12,endpoint=False)
    dirs=[unit([-.30,.96,.30])]+[unit([-.30+.45*np.cos(a),.96,.30+.40*np.sin(a)]) for a in angles]
    for d in dirs:
        proj=vv-((vv@n-height)/(d@n))[:,None]*d;uv=np.c_[proj@ex,proj@ez]
        pix=np.rint((uv-[xmin,zmin])*[(N-1)/(xmax-xmin),(M-1)/(zmax-zmin)]).astype(np.int32);mask=np.zeros((M,N),np.uint8)
        for tri in pix[ff]:cv2.fillConvexPoly(mask,tri,255,lineType=cv2.LINE_8)
        shadow+=mask.astype(np.float32)/(255*len(dirs))
    shadow=gaussian_filter(shadow,(2.2,2.8));base*=1-.65*shadow[...,None];rgb=np.uint8(np.clip(base*255,0,255))
    im=vtk.vtkImageData();im.SetDimensions(N,M,1);im.GetPointData().SetScalars(numpy_to_vtk(rgb.reshape(-1,3),deep=True,array_type=vtk.VTK_UNSIGNED_CHAR));tex=vtk.vtkTexture();tex.SetInputData(im);tex.InterpolateOn()
    p=vtk.vtkPlaneSource();p.SetOrigin(*(origin+xmin*ex+zmin*ez));p.SetPoint1(*(origin+xmax*ex+zmin*ez));p.SetPoint2(*(origin+xmin*ex+zmax*ez));p.Update();a=actor(p.GetOutput(),(1,1,1));a.SetTexture(tex);a.GetProperty().LightingOff();return a

def surface_actor(p,s,v,n):
    """Pastel material variation defined in object coordinates, not a texture."""
    f1,_=s.loop.fg(v);f2,_=s.bridge.fg(v)
    back=np.clip(.5+.5*(f1-f2)/s.blend,0,1)
    # Silvery lavender in the recessed web and near the inward-facing walls.
    side=np.clip(1-n[:,2],0,1)
    mix=np.clip(.58*back+.25*side,0,.74)
    ivory=np.array([.98,.91,.95]);lavender=np.array([.80,.69,.94]);rgb=(1-mix[:,None])*ivory+mix[:,None]*lavender
    ar=numpy_to_vtk(np.uint8(np.clip(rgb*255,0,255)),deep=True,array_type=vtk.VTK_UNSIGNED_CHAR);ar.SetName('material');p.GetPointData().SetScalars(ar)
    a=actor(p,(1,1,1),.39,.65,.065,45);a.GetMapper().SetColorModeToDirectScalars();a.GetMapper().ScalarVisibilityOn();return a

def render(a):
    print('Constructing the implicit 3D surface...',flush=True);s=Surface(a.mesh_resolution);v,f,n=s.mesh()
    ee=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);un,count=np.unique(ee,axis=0,return_counts=True);chi=len(v)-len(un)+len(f)
    if np.any(count!=2) or chi!=-2:raise RuntimeError(f'Mesh is not closed genus two: chi={chi}, invalid edges={sum(count!=2)}')
    fn=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    if np.mean(np.sum(fn*n[f[:,0]],axis=1))<0:f=f[:,[0,2,1]]
    p=polydata(v,f,n)
    conn=vtk.vtkPolyDataConnectivityFilter();conn.SetInputData(p);conn.SetExtractionModeToAllRegions();conn.Update()
    if conn.GetNumberOfExtractedRegions()!=1:raise RuntimeError("Surface has multiple connected components")
    print(f'Watertight mesh: {len(v):,} vertices; {len(f):,} triangles; Euler characteristic {chi}.',flush=True)
    gs=build_graphs(s,a.seed);res=np.max(np.abs(s.fg(np.concatenate([g['surface_paths'].reshape(-1,3) for g in gs]))[0]));print(f'{len(set(g["kind"] for g in gs))} distinct graph motifs; {len(gs)} graph charts, {sum(len(g["nodes"]) for g in gs)} vertices, {sum(len(g["edges"]) for g in gs)} edges, maximum |F|={res:.2e}.',flush=True)
    if a.export_mesh:
        wr=vtk.vtkXMLPolyDataWriter();wr.SetFileName(str(a.export_mesh));wr.SetInputData(p);wr.Write()
    if a.export_graphs:Path(a.export_graphs).write_text(json.dumps([{k:(x.tolist() if isinstance(x,np.ndarray) else x) for k,x in g.items()} for g in gs],indent=2))
    ren=vtk.vtkRenderer();ren.SetBackground(.13,.12,.165);ren.AutomaticLightCreationOff();ren.AddActor(surface_actor(p,s,v,n))
    for g in gs:ren.AddActor(actor(graph_geometry(g),g['color'],.27,.73,.46,42))
    print('Computing cast shadows...',flush=True);ren.AddActor(floor(v,f))
    for pos,col,intensity in [((-3.8,6.5,6.),(1,.94,.88),.73),((4,2,6),(.92,.91,1),.38),((1,5,-4),(.78,.64,1),.50),((-2,-3,4),(.76,.67,1),.38)]:
        l=vtk.vtkLight();l.SetLightTypeToSceneLight();l.SetPosition(*pos);l.SetFocalPoint(0,0,0);l.SetColor(*col);l.SetIntensity(intensity);l.PositionalOff();ren.AddLight(l)
    cam=ren.GetActiveCamera();cam.SetFocalPoint(0,0,0);cam.SetPosition(0,0,12);cam.SetViewUp(0,1,0);cam.ParallelProjectionOn();cam.SetParallelScale(H/(2*UNIT)*1.012)
    if a.perspective:
        cam.ParallelProjectionOff();cam.SetViewAngle(np.rad2deg(2*np.arctan(H/(2*UNIT)*1.012/12)))
    cam.Azimuth(a.azimuth);cam.Elevation(a.elevation);cam.OrthogonalizeViewUp();ren.ResetCameraClippingRange()
    ren.SetUseSSAO(True);ren.SetSSAORadius(.27);ren.SetSSAOBias(.006);ren.SetSSAOKernelSize(64);ren.SetSSAOBlur(True)
    win=vtk.vtkRenderWindow();win.SetOffScreenRendering(1);win.SetMultiSamples(0);win.AddRenderer(ren);ren.UseFXAAOn();width=a.width*a.supersample;height=round(width*H/W);win.SetSize(width,height)
    print('Rendering camera projection...',flush=True);win.Render();cap=vtk.vtkWindowToImageFilter();cap.SetInput(win);cap.ReadFrontBufferOff();cap.Update()
    image=Image.fromarray(vtk_to_numpy(cap.GetOutput().GetPointData().GetScalars()).reshape(height,width,3)[::-1]);win.Finalize()
    if a.supersample>1:image=image.resize((a.width,round(a.width*H/W)),Image.Resampling.LANCZOS)
    image.save(a.output);print('Saved '+str(a.output),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='twistor.png');p.add_argument('--width',type=int,default=1800);p.add_argument('--supersample',type=int,default=1,choices=[1,2]);p.add_argument('--mesh-resolution',type=int,default=300);p.add_argument('--azimuth',type=float,default=0);p.add_argument('--elevation',type=float,default=0);p.add_argument('--seed',type=int,default=17);p.add_argument('--perspective',action='store_true',help='Use a perspective camera instead of orthographic');p.add_argument('--export-mesh',type=Path);p.add_argument('--export-graphs',type=Path);render(p.parse_args())

if __name__=='__main__':main()
