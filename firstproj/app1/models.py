from django.db import models

# Create your models here.

class app1(models.Model):
    prod_no= models.IntegerField()
    prod_name= models.CharField(max_length=50)
    prod_price= models.FloatField() 