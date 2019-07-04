from django.shortcuts import render, redirect
#from django.core.urlresolvers import reverse_lazy
#from django.contrib.auth import autheticate, login
from django.contrib.auth import authenticate , login, logout
from django.http import HttpResponse
from django.views.generic import View
from .forms import *
from django.contrib.auth.forms import UserCreationForm
from .models import  Uploadpics,Styles
from style.settings import MEDIA_ROOT,BASE_DIR
from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import ImageGrab, ImageOps
import PIL.Image
import numpy as np
import json
from django.views.decorators.csrf import csrf_exempt
import os


# importing for style transfer-----------------------------------------------------------------------
import os.path

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import time
import functools
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from keras.preprocessing.image import img_to_array
tf.enable_eager_execution()
#print("Eager execution: {}".format(tf.executing_eagerly()))
def load_img(path_to_img):
	
	max_dim = 1080
	img = Image.open(path_to_img)
	long = max(img.size)
	scale = max_dim/long
	img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
  
	img = kp_image.img_to_array(img)
  
  # We need to broadcast the image array such that it has a batch dimension 
	img = np.expand_dims(img, axis=0)
	print("doneeeeee")
	return img

def imshow(img, title=None):

  # Remove the batch dimension
	out = np.squeeze(img, axis=0)
  # Normalize for display 
	out = out.astype('uint8')
	plt.imshow(out)
	if title is not None:
		plt.title(title)
		plt.imshow(out)



def load_and_process_img(path_to_img):
	
	
	img = load_img(path_to_img)
	img = tf.keras.applications.vgg16.preprocess_input(img)
	return img


def deprocess_img(processed_img):
	
	x = processed_img.copy()
	if len(x.shape) == 4:
		
		x = np.squeeze(x, 0)
	assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
							 "dimension [1, height, width, channel] or [height, width, channel]")
	if len(x.shape) != 3:
		raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessiing step
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68
	x = x[:, :, ::-1]

	x = np.clip(x, 0, 255).astype('uint8')
	return x


vgg = tf.keras.applications.vgg16.VGG16(
	weights="imagenet",
	include_top=False)





content_layers = ['block2_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv2','block2_conv2','block3_conv3','block4_conv3','block5_conv3' ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def get_model():
	
 
  # Load our model.
	vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
	vgg.trainable = False
  # Get output layers corresponding to style and content layers 
	style_outputs = [vgg.get_layer(name).output for name in style_layers]
	content_outputs = [vgg.get_layer(name).output for name in content_layers]
	model_outputs = style_outputs + content_outputs
  # Build model 
	return models.Model(vgg.input, model_outputs)




def get_content_loss(base_content, target):
	return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
	
	
  # We make the image channels first 
	channels = int(input_tensor.shape[-1])
	a = tf.reshape(input_tensor, [-1, channels])
	n = tf.shape(a)[0]
	gram = tf.matmul(a, a, transpose_a=True)
	return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
	height, width, channels = base_style.get_shape().as_list()
	gram_style = gram_matrix(base_style)
  
	return tf.reduce_mean(tf.square(gram_style - gram_target))
	
	



def get_feature_representations(model, content_path, style_path):
	
 
	content_image = load_and_process_img(content_path)
	style_image = load_and_process_img(style_path)
  
  # batch compute content and style features
	style_outputs = model(style_image)
	content_outputs = model(content_image)
  
  
  # Get the style and content feature representations from our model  
	style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
	content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
	return style_features, content_features

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
	
 
	style_weight, content_weight = loss_weights
  
  # Feed our init image through our model. This will give us the content and 
  # style representations at our desired layers. Since we're using eager
  # our model is callable just like any other function!
	model_outputs = model(init_image)
  
	style_output_features = model_outputs[:num_style_layers]
	content_output_features = model_outputs[num_style_layers:]
  
	style_score = 0
	content_score = 0
# Accumulate style losses from all layers
  # Here, we equally weight each contribution of each loss layer
	weight_per_style_layer = 1.0 / float(num_style_layers)
	for target_style, comb_style in zip(gram_style_features, style_output_features):
		style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
	
  # Accumulate content losses from all layers 
	weight_per_content_layer = 1.0 / float(num_content_layers)
	for target_content, comb_content in zip(content_features, content_output_features):
		content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
  
	style_score *= style_weight
	content_score *= content_weight

  # Get total loss
	loss = style_score + content_score 
	return loss, style_score, content_score

def compute_grads(cfg):
	
	with tf.GradientTape() as tape: 
		all_loss = compute_loss(**cfg)
  # Compute gradients wrt input image
	total_loss = all_loss[0]
	return tape.gradient(total_loss, cfg['init_image']), all_loss

import IPython.display

def run_style_transfer(content_path, 
					   style_path,
					   num_iterations=10,
					   content_weight=1, 
					   style_weight=1e3): 
	
  # We don't need to (or want to) train any layers of our model, so we set their
  # trainable to false. 
	model = get_model() 
	for layer in model.layers:
		layer.trainable = False
  
  # Get the style and content feature representations (from our specified intermediate layers) 
	style_features, content_features = get_feature_representations(model, content_path, style_path)
	gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
  # Set initial image
	init_image = load_and_process_img(content_path)
	init_image = tfe.Variable(init_image, dtype=tf.float32)
  # Create our optimizer
	opt = tf.train.AdamOptimizer(learning_rate=20, beta1=0.99, epsilon=1e-1)
  
  # Store our best result
	best_loss, best_img = float('inf'), None
  # Create a nice config 
	loss_weights = (style_weight, content_weight)
	cfg = {
	  'model': model,
	  'loss_weights': loss_weights,
	  'init_image': init_image,
	  'gram_style_features': gram_style_features,
	  'content_features': content_features
  }
	
  # For displaying
	num_rows = 2
	num_cols = 5
	display_interval = num_iterations/(num_rows*num_cols)
	start_time = time.time()
	global_start = time.time()
	norm_means = np.array([103.939, 116.779, 123.68])
	min_vals = -norm_means
	max_vals = 255 - norm_means   
  
	imgs = []
	with tf.device("/gpu:0"):
		
		for i in range(num_iterations):
			grads, all_loss = compute_grads(cfg)
			loss, style_score, content_score = all_loss
			opt.apply_gradients([(grads, init_image)])
		#clipped = tf.clip_by_value(init_image, min_vals, max_vals)
		#init_image.assign(clipped)
			end_time = time.time() 

#             print(". ", end="") # Fo tracking progress

			if loss < best_loss:
				
		  # Update best loss and best image from total loss. 
				best_loss = loss
				best_img = deprocess_img(init_image.numpy())
			if i % display_interval== 0:
				start_time = time.time()

		  # Use the .numpy() method to get the concrete numpy array
				plot_img = init_image.numpy()
				plot_img = deprocess_img(plot_img)
				imgs.append(plot_img)
#           IPython.display.clear_output(wait=True)
#           IPython.display.display_png(Image.fromarray(plot_img))
#           print('Iteration: {}'.format(i))        
#           print('Total loss: {:.4e}, ' 
#                 'style loss: {:.4e}, '
#                 'content loss: {:.4e}, '
#                 'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
#   print('Total time: {:.4f}s'.format(time.time() - global_start))
#   IPython.display.clear_output(wait=True)
#   plt.figure(figsize=(14,4))
#   for i,img in enumerate(imgs):
#       plt.subplot(num_rows,num_cols,i+1)
#       plt.imshow(img)
#       plt.xticks([])
#       plt.yticks([])
	  
	return best_img, best_loss 



plt.figure(figsize=(10,10))

# content = load_img(content_path).astype('uint8')
# style = load_img(style_path).astype('uint8')

# # plt.subplot(1, 2, 1)
# # imshow(content, 'Content Image')

# # plt.subplot(1, 2, 2)
# # imshow(style, 'Style Image')
# # plt.show()

# best, best_loss = run_style_transfer(content_path, 
#                                      style_path, num_iterations=10)


def show_results(best_img, content_path, style_path, show_large_final=True):

	
	plt.figure(figsize=(10, 5))
	plt.imshow(best_img)
	# plt.show()

	# content = load_img(content_path) 
	# style = load_img(style_path)

	# plt.subplot(1, 2, 1)
	# imshow(content, 'Content Image')

	# plt.subplot(1, 2, 2)
	# imshow(style, 'Style Image')

	# if show_large_final: 
	# 	plt.figure(figsize=(10, 10))

	# 	plt.imshow(best_img)
	# 	plt.title('Output Image')
	# 	plt.show()
	
















# style transfer end------------------------------------------------------------------------------------
# Create your views here.


def index(request):
	return render(request, 'style_transfer/home.html', {})

def main_pg(request):
	return render(request, 'style_transfer/main_page.html', {})

def about_project(request):
	return render(request, 'style_transfer/about_project.html', {})

def about_team(request):
	return render(request, 'style_transfer/about_team.html', {})

def instruction(request):
	return render(request, 'style_transfer/instructions.html', {})

def upload_content(request):
	# Handle file upload
	if request.method == 'POST':
		form = Upload_content_style(request.POST or None, request.FILES or None)
		print(request.FILES)
		if form.is_valid():

			print("asdfasdfgb")
			ins=form.cleaned_data['Style_num']
			# content_path1=request.FILES['Contentfile'] or None
			# print(content_path1)
			print (ins)
			# print(content_path1)
			# newdoc = Upload_content_style()
			instance=form.save(commit=False)

			instance.save()
			ins2= Uploadpics.objects.get(id=instance.id)
			if(ins2.Url_field==None):

				cont=ins2.Contentfile.path

			else:
				cont="nothing"
			dood_url=ins2.Url_field
			print("1111111111111111"+str(dood_url))
			print("222222222222222222"+str(cont))
			my_mod=Styles.objects.get(Style_num=ins)
			print(my_mod)
			style_path = my_mod.Stylefile.path
			print(style_path)


			# form.save(commit=False)

			# call style transfer code----------------------------
			#content_path = 'niki.jpg'
			#style_path = 'style3.jpg'


			# plt.figure(figsize=(10,10))
			tempo=""
			if(str(dood_url)=='None'):
				
				content = load_img(cont).astype('uint8')
				style = load_img(style_path).astype('uint8')
				best, best_loss = run_style_transfer(cont, 
												 style_path, num_iterations=10)
				y=np.random.randn()
				tempo="out"+str(y)+".jpg"
				show_results(best, cont, style_path)
				image_path = os.path.join(MEDIA_ROOT,"saved_images",tempo)
				# cwd = os.getcwd()

				img_path2=os.path.join(BASE_DIR,'media','saved_images',tempo)
				print(BASE_DIR)
				print(image_path+"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
				print(img_path2)

				Image.fromarray(best).save(img_path2, "JPEG", quality=80, optimize=True, progressive=True)
				img_path2='/media/'+'saved_images/'+tempo



			else:
				content = load_img(dood_url).astype('uint8')
				style = load_img(style_path).astype('uint8')
				best, best_loss = run_style_transfer(dood_url, 
												 style_path, num_iterations=10)
				y=np.random.randn()
				tempo="out"+str(y)+".jpg"
				show_results(best, dood_url, style_path)
				img_path2=os.path.join(BASE_DIR,'media','saved_images',tempo)
				image_path = os.path.join(MEDIA_ROOT,"saved_images",tempo)
				print(image_path+"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
				print(img_path2)
				Image.fromarray(best).save(img_path2, "JPEG", quality=80, optimize=True, progressive=True)
				img_path2='/media/'+'saved_images/'+tempo
				

			request.session["path_img"]=img_path2


			# style = load_img(style_path).astype('uint8')
			# # print("doooooooooooonnnnnnnneeeeeeeeeeeeeeeeeeeee")
			# # plt.subplot(1, 2, 1)
			# # imshow(content, 'Content Image')

			# # plt.subplot(1, 2, 2)
			# # imshow(style, 'Style Image')
			# # plt.show()

			# best, best_loss = run_style_transfer(cont, 
			# 									 style_path, num_iterations=10)
			# y=np.random.randn()

			# show_results(best, cont, style_path)
			# Image.fromarray(best).save("out"+str(y)+".jpg", "JPEG", quality=80, optimize=True, progressive=True)
			# print("hoggyyyyyyyyyyaaaaaaaaaaaaa")
			# #call style transfer here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			
			return redirect("saved")
			
	else:
		form = Upload_content_style() # A empty, unbound form
		context={
		"form": form,
		"list": Styles.objects.all(),
		}
		return render(request, 'style_transfer/upload3.html', context)


def saved(request):
	context={
	"path_img": request.session["path_img"]
	}
	return render(request, 'style_transfer/saved_img.html',context)

def login_user(request):
	if request.user.is_authenticated:
		logout(request)

	if request.method=='POST':
		form=login_form(request.POST)

		if form.is_valid():
			username=form.cleaned_data.get("username")
			password=form.cleaned_data.get("password")
			user = authenticate(username=username, password=password)
			if user is None:
				form1=login_form()
				context={
					"form":form1,
				}
				return render(request,'style_transfer/login.html', context)
			# print("Brook was here")
			login(request, user)
			return redirect('upload_content')

	form=login_form()
	context={
	"form":form,
	}
	return render(request,'style_transfer/login.html', context)


from django.contrib import messages 
def UserFormView(request):
	if request.method == "POST":
		form=UserForm(request.POST)
		print(request.POST["username"])
		if form.is_valid():
			form.save()
			return redirect("upload_content")
		print("invalid form")
		messages.error(request, "Error")
		form=UserForm()
		context={
		"form":form,
		}
		return render(request,"style_transfer/signup.html",context)
	print("invalid post")
	form=UserForm()
	context={
	"form":form,
	}
	return render(request,"style_transfer/signup.html",context)



@csrf_exempt
def draw_doodle(request):
	# print("hiiiiiii")
	class Paint(object):

		DEFAULT_PEN_SIZE = 10.0
		DEFAULT_COLOR = 'black'

		

		def __init__(self):
			self.root = Tk()
			#self.root.after(0, self.root.focus_force)
			self.width = 900
			self.height = 640
			self.pen_button = Button(self.root, text='pen', command=self.use_pen)
			self.pen_button.grid(row=0, column=0)
			
			self.save_button = Button(self.root, text='save', command=self.save_and_quit)
			self.save_button.grid(row=0, column=1)
	#         self.brush_button = Button(self.root, text='brush', command=self.use_brush)
	#         self.brush_button.grid(row=0, column=1)

			self.color_button = Button(self.root, text='color', command=self.choose_color)
			self.color_button.grid(row=0, column=2)

			self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
			self.eraser_button.grid(row=0, column=3)

			self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
			self.choose_size_button.grid(row=0, column=4)
			
			

			self.c = Canvas(self.root, bg='white', width=self.width, height=self.height)
			self.c.grid(row=1, columnspan=6)
			
			self.setup()
			self.root.iconify()
			self.root.deiconify()
			self.root.mainloop()

		def setup(self):
			self.old_x = None
			self.old_y = None
			self.line_width = self.choose_size_button.get()
			self.color = self.DEFAULT_COLOR
			self.eraser_on = False
			self.active_button = self.pen_button
			self.c.bind('<B1-Motion>', self.paint)
			self.c.bind('<ButtonRelease-1>', self.reset)
		def saveas(self):
			x=np.random.randn()
			filename="image"+str(x)
			
			#savename = 'im_{0:0>6}'.format(i)
			cwd = os.getcwd()
			ImageGrab.grab(()).save(cwd+'\\media\\doodled\\'+filename + '.jpg')
			fp = open(cwd+'\\media\\doodled\\'+filename+".jpg","rb")
			img = PIL.Image.open(fp)
			border = (250, 250, 200, 10)
			fname= "img"+str(x)# left, up, right, bottom
			ImageOps.crop(img, border).save(cwd+'\\media\\doodled\\' +fname+".jpg")
			self.root.destroy()
			# instance.doodle = "Z:/style/style_transfer/media/doodled" +fname+".jpg"
			response["doodle_url"]=cwd+'\\media\\doodled\\' +fname+".jpg"
		
		
		def use_pen(self):
			self.activate_button(self.pen_button)

		def save_and_quit(self):
			self.activate_button(self.save_button)
			self.saveas()
			#show_entry_fields()
			#quit()


		def choose_color(self):
			self.eraser_on = False
			self.color = askcolor(color=self.color)[1]

		def use_eraser(self):
			self.activate_button(self.eraser_button, eraser_mode=True)

		def activate_button(self, some_button, eraser_mode=False):
			self.active_button.config(relief=RAISED)
			some_button.config(relief=SUNKEN)
			self.active_button = some_button
			self.eraser_on = eraser_mode

		def paint(self, event):
			self.line_width = self.choose_size_button.get()
			paint_color = 'white' if self.eraser_on else self.color
			if self.old_x and self.old_y:
				self.c.create_line(self.old_x, self.old_y, event.x, event.y,
								   width=self.line_width, fill=paint_color,
								   capstyle=ROUND, smooth=TRUE, splinesteps=36)
			self.old_x = event.x
			self.old_y = event.y

		def reset(self, event):
			self.old_x, self.old_y = None, None
			
		

	#     def save_and_quit():
	#         self.activate_button(self.save_button)

	#         self.save('Z:/style/my.jpg', 'JPEG')
	#         #show_entry_fields()
	#         root.quit()
	response={}
	Paint()
	response_data = json.dumps(response)
	return HttpResponse(response_data, content_type="application/json") 


	
	







# Create your views here.
