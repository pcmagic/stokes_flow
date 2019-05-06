n = 10; 
x = linspace(-0.5, 0.5, 56); 
f1 = @(x) (exp(x*n)-1)./(2*(exp(0.5*n)-1));
f2 = @(x) log(2*(exp(0.5/n)-1).*x+1).*n;

y1 = sign(x).*f1(abs(x))+0.5;
y2 = sign(x).*f2(abs(x))+0.5;
y = (y1*n+y2/n)/2;
plot(x, y);