```c++

int vib_3 = 3;
int vib_5 = 5;
int vib_6 = 6; // 압력센서 1, 3번 방향(우측)에 삽입할 진동모터

int vib_9 = 9;
int vib_10 = 10;
int vib_11 = 11; // 압력센서 0, 2번 방향(좌측)에 삽입할 진동모터

int reset_btn = 7;  // 초기화버튼
int red_btn = 1/* 핀번호를 입력하세요 */; // AI 학습 버튼 1
int green_btn = 2/* 핀번호를 입력하세요 */; // AI 학습 버튼 2

int standard_x; 
int standard_y; // 무게중심 초기화 정보 저장할 변수

void setup(){
  Serial.begin(9600);

  pinMode(vib_3,OUTPUT);
  pinMode(vib_5,OUTPUT);
  pinMode(vib_6,OUTPUT);

  pinMode(vib_9,OUTPUT);
  pinMode(vib_10,OUTPUT);
  pinMode(vib_11,OUTPUT);

  pinMode(red_btn,INPUT_PULLUP);
  pinMode(green_btn,INPUT_PULLUP);

  standard_x = 0;
  standard_y = 0;

  Serial.println("CLEARDATA");
  Serial.println("LABEL,StnX,StnY,X0,Y0,X,Y,GBN");
}

void loop(){

  int GoodBadNull = 0; // AI학습용
  
  // 0번 압력센서는 A0핀에
  // 1번 압력센서는 A1핀에
  // 2번 압력센서는 A2핀에
  // 3번 압력센서는 A3핀에 -> 압력센서를 아날로그핀에 할당
  int press_0 = analogRead(A0)/9;
  int press_1 = analogRead(A1)/9;
  int press_2 = analogRead(A2)/4;
  int press_3 = analogRead(A3)/4; // 각 압력센서에서 입력받은 값을 press 변수에 저장

  int xbar_0 = press_1 + press_3 - press_0 - press_2;
  int ybar_0 = press_2 + press_3 - press_0 - press_1; // 평면좌표 설정하여 무게중심의 좌표 구함

  int xbar = xbar_0 - standard_x;
  int ybar = ybar_0 - standard_y;

    if(digitalRead(reset_btn) == 0){ // 버튼이 눌렸다면
      Serial.println("Get the Right Posture Plz"); // 자세를 바르게 해주세요
      delay(1000);
      Serial.println("5");
      delay(1000);
      Serial.println("4");
      delay(1000);
      Serial.println("3");
      delay(1000);
      Serial.println("2");

      press_0 = analogRead(A0)/9;
      press_1 = analogRead(A1)/9;
      press_2 = analogRead(A2)/4;
      press_3 = analogRead(A3)/4; // 각 압력센서에서 입력받은 값을 press 변수에 저장

      xbar_0 = press_1 + press_3 - press_0 - press_2;
      ybar_0 = press_2 + press_3 - press_0 - press_1; // 평면좌표 설정하여 무게중심의 좌표 구함

      delay(1000);
      Serial.println("1"); // 카운트다운
      standard_x = xbar_0;
      standard_y = ybar_0; // 정자세 무게중심을 좌표값으로 받음

      delay(2000);
      Serial.println("Setup Complete"); // 설정 완료
      delay(300);
    }

    // 평면의 전체 크기를 600 * 600 으로 설정
    // 7번 8번 9번칸
    // 4번 5번 6번칸
    // 1번 2번 3번칸 <- 좌표상의 구성
/*
    if(xbar>= 25 && ybar>= 25){ // 9번칸
      state = 9;
      delay(300);
      analogWrite(vib_6,0);
    }
    if((xbar> -25 && xbar< 25) && ybar>= 25){ // 8번칸
      state = 8;
      delay(300);
    }
    if(xbar<= -25 && ybar>= 25){ // 7번칸
      state = 7;
      delay(300);
      analogWrite(vib_9,0);
    }
    if(xbar>= 25 && (ybar > -25 && ybar < 25)){ // 6번칸
      state = 6;
      delay(300);
      analogWrite(vib_5,0);
    }
    if((xbar> -25 && xbar< 25) && (ybar > -25 && ybar < 25)){ // 5번칸
      state = 5;
      delay(300);
    }
    if(xbar<= -25 && (ybar > -25 && ybar < 25)){ // 4번칸
      state = 4;
      delay(300);
      analogWrite(vib_10,0);
    }
    if(xbar>= 25 && ybar<= -25){ // 3번칸
      state = 3;
      delay(300);
      analogWrite(vib_3,0);
    }
    if((xbar> -25 && xbar< 25) && ybar<= -25){ // 2번칸
      state = 2;
      delay(300);
    }
    if(xbar<= -25 && ybar<= -25){ // 1번칸
      state = 1;
      delay(300);
      analogWrite(vib_11,0);
    }
*/
    if(digitalRead(green_btn) == 0){ // 좋은자세
      GoodBadNull = 2;
    }
    else if(digitalRead(red_btn) == 0){ // 나쁜자세
      GoodBadNull = 1;
    }
    else{ // 없어용
      GoodBadNull = 0;
    }



    Serial.print("DATA,");
    Serial.print(standard_x);
    Serial.print(",");
    Serial.print(standard_y);
    Serial.print(",");
    Serial.print(xbar_0);
    Serial.print(",");
    Serial.print(ybar_0);
    Serial.print(",");
    Serial.print(xbar);
    Serial.print(",");
    Serial.print(ybar);
    Serial.print(",");
    Serial.println(GoodBadNull);

    delay(300);

  
}

```
