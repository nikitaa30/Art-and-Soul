from django.db import models



def upload_location(instance, filename):
	return "%s/%s" %(instance.id, filename)

# class Document(models.Model):
#     docfile = models.FileField(upload_to = 'contentupload/', null = True, blank=True)
# Create your models here.
class Uploadpics(models.Model):
	Contentfile= models.ImageField(upload_to ='contentupload/', null = True, blank=True,verbose_name="")
	Url_field= models.CharField(max_length=100, null = True, blank=True)
	Style_num= models.IntegerField(default=1)
	class Meta:
		ordering = ["-pk"]

class Styles(models.Model):
	Style_num= models.IntegerField(default=1)
	Stylefile = models.ImageField(upload_to = 'stylefile/', null = True, blank=True)
