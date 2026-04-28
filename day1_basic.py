##python基础
class Config:
    def __init__(self,model,temperature,api_key):
        self.model=model
        self.temperature = temperature
        self.api_key = api_key
    @property
    def info(self):
        return f"模型:{self.model},温度:{self.temperature}"

## 装饰器
def log_call(func):
    def wrapper(*args,**kwargs):
        print(f">>调用函数{func.__name__}")
        result = func(*args,**kwargs)
        print(f">>返回结果{result}")
        return result
    return wrapper
## 推导式
squares =[i**2 for i in range(1,11)]

if __name__ == "__main__":
    config = Config("GLM-5","0.7","bce-v3/ALTAK-Es14qJEZtcSOlQMVudAPZ/ae25fd619724058f96864b7f37387e28932422aa")
    print(config.info)
    @log_call
    def add(a,b):
        return a+b
    result = add(3,5)
    print(f"平方数{squares}")
