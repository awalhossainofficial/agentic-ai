from typing import Optional
from pydantic import BaseModel,EmailStr,Field

class Student(BaseModel):
    name:str = "Rahul"
    age:Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=4, default=5, description="A decimal value representing the cgpa of the student")


new_stu = {'age':32, 'email':'abc@gmail.com'}

student = Student(**new_stu)

print(student)