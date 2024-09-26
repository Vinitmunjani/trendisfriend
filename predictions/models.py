from django.db import models
from django.utils import timezone 
from django.contrib.auth.models import User

class StockPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)
    ticker = models.CharField(max_length=10)
    last_price = models.FloatField()
    predicted_price = models.FloatField()
    target_price = models.FloatField()
    movement = models.FloatField()
    volume = models.IntegerField(default=0)
    volume_change = models.FloatField(default=0)
    timestamp = models.CharField(max_length=100)
    
    def __str__(self) -> str:
        return f"{self.ticker} - {self.timestamp}"


class Option(models.Model):
    stock_prediction = models.ForeignKey(StockPrediction, on_delete=models.CASCADE, related_name='options')
    option_strike = models.CharField(max_length=100)
    option_type = models.CharField(max_length=4)  # 'CALL' or 'PUT'
    security_id = models.CharField(max_length=20)
    lot_size = models.IntegerField()

    def __str__(self):
        return f"{self.option_strike} - {self.security_id}"


class Order(models.Model):
    option = models.ForeignKey(Option, on_delete=models.CASCADE, related_name='orders')
    order_id = models.CharField(max_length=100, unique=True,null=True,blank=True)
    option_strike = models.CharField(max_length=100)
    price = models.FloatField()
    stoploss = models.FloatField()  # This should be 5% of the price
    target = models.FloatField()    # This should be 10% of the price
    order_time = models.DateTimeField(default=timezone.now)
    lot_size = models.IntegerField()
    
    order_value = models.FloatField()
    profit = models.FloatField(null=True, blank=True, default=0)
    loss = models.FloatField(null=True, blank=True, default=0)
    
    def save(self, *args, **kwargs):
        # Automatically set stoploss and target based on the price before saving
        if not self.stoploss:
            self.stoploss = self.price * 0.95
        if not self.target:
            self.target = self.price * 1.10
        
        if not self.order_value:
            self.order_value = self.price * self.lot_size
        
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Order {self.order_id} for {self.option_strike} at {self.price}"
