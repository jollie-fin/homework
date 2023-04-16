'''
Created on 16 janv. 2011

@author: elie gedeon
'''

#l'idee d'implementer le programme avec une image est de pouvoir plus facilement etendre le programme a
#trois ou plus joueurs

#z q s d joueur bleu
#haut gauche bas droite 
import Tkinter
import Image, ImageTk

def button_click_exit_mainloop (event):
    event.widget.quit() # this will cause mainloop to unblock.

#Width
W = 640
#Height
H = 480
#Resolution
R = 5


p = Image.new("RGB",[W,H])
#pix est un accesseur a la zone memoire
pix = p.load()

#trace active des joueurs
trace = ((255,0,1),(1,128,255))
#point representant les joueurs
moto = ((255,255,255),(255,255,255))
#utilise par les fonctions recursives 
test = (128,128,128)
#utilise par les fonctions recursives pour signaler ce qui n'appartient pas au joueur
nappartient = (35,76,12)
#zone achevees
remplissage = ((255,0,0),(0,128,255))
#rien
noir = (0,0,0)
#pourtour
neutre = (64,64,64)

#score courant
score = [0,0]

#position courante
position = [(0+R/2,H/2/R*R+R/2),((W-1)/R*R+R/2,H/2/R*R+R/2)]
#direction courante
direction = [(0,0),(0,0)]
#etat courante
etat = [0,0]
 
#initialise l'image
for i in range(W):
    for j in range(H):
        pix[i,j]=noir

   
for i in range(H):
    for j in range(R):
        pix[j,i]=neutre
        pix[W-1-j,i]=neutre
   
for i in range(W):
    for j in range(R):
        pix[i,j]=neutre
        pix[i,H-1-j]=neutre


position_a_inspecter = []
position_a_inspecter_global = []
remplissage_possible = True
nombre_remplissage = 0
nombre_remplissage_global = 0
position_ennemie = None
trace_ennemie = None
couleur = None


#une vraie fonction recursive aurait ete plus claire, mais la pile
#d'appels est tres limitee; on la simule avec une pile: position_a_inspecter

#remplace toutes les cases qui contiennent test par avecquoi
def fixe_remplissage_aux():
    global position_a_inspecter
    global pix
    global couleur
    (i,j) = position_a_inspecter.pop()
    if (pix[i,j] == test):
        pix[i,j] = couleur
        position_a_inspecter.extend([(i-R,j),(i+R,j),(i,j-R),(i,j+R)])

def fixe_remplissage(x,y,avecquoi):
    global position_a_inspecter
    global couleur
    couleur = avecquoi
    position_a_inspecter = [(x,y)]
    while (position_a_inspecter<>[]):
        fixe_remplissage_aux()

#tente de remplir la zone centree en x,y, et renvoie le nombre de cases remplies 
def tente_remplir_aux():
    global nombre_remplissage
    global remplissage_possible
    global position_a_inspecter
    global pix
    global couleur
    (i,j)= position_a_inspecter.pop()

    if (i,j) == position[1-couleur]:
        remplissage_possible = False
    elif (pix[i,j] == noir):
        nombre_remplissage += 1
        pix[i,j]=test
        position_a_inspecter.extend([(i-R,j),(i+R,j),(i,j-R),(i,j+R)])
    elif (pix[i,j] == trace[1-couleur]):
        remplissage_possible = False

def tente_remplir(x,y,c):
    global remplissage_possible
    global position_a_inspecter
    global nombre_remplissage
    global position_ennemie
    global trace_ennemie
    global couleur
    couleur = c
    position_ennemie = position[1-c]
    trace_ennemie = trace[1-c]
    remplissage_possible = True
    position_a_inspecter = [(x,y)]
    nombre_remplissage = 0
    while (position_a_inspecter<>[]):
        tente_remplir_aux()
    
    if (remplissage_possible):
        fixe_remplissage(x,y,remplissage[c])
    else:
        fixe_remplissage(x,y,nappartient)
        nombre_remplissage = 0
    return nombre_remplissage
    
#efface les traces de nappartient    
def nettoie_test():
    global pix
    for i in range(0,W):
        for j in range(0,H):
            if (pix[i,j] == nappartient):
                pix[i,j] = noir

#parcours recursivement la trace du joueur quand elle est fermee, et appel tente remplir de part et d'autre
def acheve_trace_aux(c):
    global position_a_inspecter_global
    global pix
    global nombre_remplissage_global    
    (i,j) = position_a_inspecter_global.pop()
    if (pix[i,j] == trace[c]):
        pix[i,j] = remplissage[c]
        nombre_remplissage_global += 1
        position_a_inspecter_global.extend([(i-R,j),(i+R,j),(i,j-R),(i,j+R)])
    elif (pix[i,j] == noir):
        nombre_remplissage_global += tente_remplir(i,j,c)

def acheve_trace(x,y,c):
    global position_a_inspecter_global
    global nombre_remplissage_global    
    position_a_inspecter_global = [(x,y)]
    nombre_remplissage_global = 0
    while (position_a_inspecter_global<>[]):
        acheve_trace_aux(c)
    nettoie_test()
    return nombre_remplissage_global

#efface la trace (quand elle a ete coupee
def detruit_trace_aux(c):
    global position_a_inspecter_global
    global pix
    global nombre_remplissage_global    
    (i,j) = position_a_inspecter_global.pop()
    if (pix[i,j] == trace[c]):
        pix[i,j] = noir
        position_a_inspecter_global.extend([(i-R,j),(i+R,j),(i,j-R),(i,j+R)])

def detruit_trace(x,y,c):
    global position_a_inspecter_global    
    position_a_inspecter_global = [(x,y)]
   
    while (position_a_inspecter_global<>[]):
        detruit_trace_aux(c)
    return nombre_remplissage_global

root = None
tkpi = None
clock = None
labels = [None,None]
label_image = None
#evenement clavier
def changedirection(event):
    x = event.keysym_num
    if x==65361:#gauche
        direction[0] = (-R,0)
    if x==65363:#droite
        direction[0] = (R,0)
    if x==65362:#haut
        direction[0] = (0,-R)
    if x==65364:#bas
        direction[0] = (0,R)
    if x==113:#gauche
        direction[1] = (-R,0)
    if x==100:#droite
        direction[1] = (R,0)
    if x==122:#haut
        direction[1] = (0,-R)
    if x==115:#bas
        direction[1] = (0,R)

def initialise_affichage():
    global root
    global clock
    global labels
    global label_image
    root = Tkinter.Tk()
    root.title("bonjour")
    root.bind("<Button>", button_click_exit_mainloop)
    root.geometry('+%d+%d' % (W,H+100))
    root.geometry('%dx%d' % (W,H+100))
    root.bind_all('<Key>', changedirection)
    clock = Tkinter.Label(  )
    clock.pack(  )
    labels[0]=Tkinter.Label(root, text="rouge " + repr(score[0]))
    labels[1]=Tkinter.Label(root, text="bleu " + repr(score[1]))
    labels[0].pack()
    labels[1].pack()
    label_image = Tkinter.Label(root)
    label_image.place(x=0,y=100,width=W,height=H)

def actualise_labels():
    labels[0].configure(text="rouge " + repr(score[0]*100/((W-2*R)/R*(H-2*R)/R)) + "%")
    labels[1].configure(text="bleu " + repr(score[1]*100/((W-2*R)/R*(H-2*R)/R)) + "%")


def attend():
    root.mainloop() # wait until user clicks the window    

def affiche():
    global p
    global root
    global tkpi
    global label_image
    actualise_labels()
    tkpi = ImageTk.PhotoImage(p)
    label_image.config(image=tkpi)

#etat 0: joueur trace sa ligne
#etat 1: trace coupee => joueur ne trace pas de ligne
#etat 2: joueur en zone neutre    
def deplace(c): #0 pour rouge, 1 pour bleu
    global pix
    global W
    global H
    
    global clock
    
    (x0,y0) = position[c]
    (dx,dy) = direction[c]
    
    x = x0+dx
    y = y0+dy
    
       
    if (x>=0) and (x<W) and (y>=0) and (y<H):
        if pix[x,y]==noir:
            if etat[c]==2:
                etat[c]=0
            if etat[c]==0:
                pix[x,y] = trace[c]
            position[c] = (x,y)
        elif pix[x,y]==trace[c]:
            etat[c]=1
            detruit_trace(x,y,c)
            position[c] = (x,y)
        elif pix[x,y]==trace[1-c]:
            detruit_trace(x,y,1-c)
            etat[1-c]=1
            if etat[c]==2:
                etat[c]=0
            if etat[c]==0:
                pix[x,y] = trace[c]    
            position[c] = (x,y)
        else:
            if etat[c]<>2:
                direction[c] = (0,0)
                if etat[c] == 0:
                    score[c] += acheve_trace(x0,y0,c)
                etat[c]=2
            position[c] = (x,y)

    else:
        direction[c] = (0,0)
        x=x0
        y=y0

def teste_fin():
    if score[0] + score[1] == (W/R-2)*(H/R-2):
        if score[0]>score[1]:
            labels[0].configure(text="rouge gagne")
        if score[0]<score[1]:
            labels[0].configure(text="bleu gagne")
        if score[0]<score[1]:
            labels[0].configure(text="egalite")
        labels[1].configure(text="")
        root.mainloop()
        quit()

#toutes les 10ms, appel affiche
def timer():
    deplace(0)
    deplace(1)
    xr,yr = position[0]
    xb,yb = position[1]
    pr = pix[xr,yr]
    pb = pix[xb,yb]
    pix[xr,yr] = moto[0]
    pix[xb,yb] = moto[1]
    affiche()
    teste_fin()
    pix[xr,yr]=pr
    pix[xb,yb]=pb
    clock.after(10,timer)


initialise_affichage()

timer()
root.mainloop()
    



