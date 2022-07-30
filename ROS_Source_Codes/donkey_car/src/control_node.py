#!/usr/bin/python
"""
Gets the offset and angle delivered by camera node and calculates the control action

Subscribes to 
    /blob/point_blob
    
Publishes commands to 
    /dkcar/control/cmd_vel    

"""
import numpy as np
import rospy
import time
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from scipy.signal import cont2discrete
from scipy.linalg import block_diag
from numpy.linalg import matrix_power
import osqp
from scipy import sparse
from i2cpwm_board.msg import Servo, ServoArray

#Constants declaration
mass=1.24
Iz=0.75
Calfa_front=1
Calfa_rear=1
lf=0.0885
lr=0.0885
Ts=0.142857
Vx=0.75
V_pwm=0.65
#Ky=1
#Kpsi=3
#Qgain=0.1
#Rgain=100
N=12
vmax=0.262
vmin=-0.262
epsilon_max_limits=1
epsilon_min_limits=-1
w_covar=0.001
w_bar=0
g_const=np.array([1,0,0,0])
Distancia_Sim = 20
Nsim=int(Distancia_Sim/(Vx*Ts));
dist_constraint1 = 3; 
dist_constraint2 = 5; 
dist_constraint3 = 8;
dist_constraint4 = 10;
dist_constraint5 = 13; 
dist_constraint6 = 15; 
distlength = 1;
PWM_Angle_Center=341


def saturate(value, min, max):
    if value <= min: return(min)
    elif value >= max: return(max)
    else: return(value)

class ControlNode():
    def __init__(self):
    #348.5
        self.Speed_PWM = 348.2
    
    	#--- Create the servos publisher, Steering and Throttle
        self.ros_pub_servo_array    = rospy.Publisher("/servos_absolute", ServoArray, queue_size=1)
        rospy.loginfo("> Servo Publisher corrrectly initialized")
        
        #--- Create a debug publisher for resulting cmd_vel
        self.ros_pub_debug_command    = rospy.Publisher("/dkcar/debug/cmd_vel", Twist, queue_size=1)
        rospy.loginfo("> Debug Publisher corrrectly initialized")
        
        #--- Create a debug publisher for the vizualisation of control restrictions
        self.safe_pub  = rospy.Publisher("/safe/safe_restriction",Point,queue_size=1) 
	self.safe_restriction = Point()
	rospy.loginfo("> Debug Publisher corrrectly initialized")
	
        #--- Create the Subscriber to Twist commands in case of driver's behavior
        self.ros_sub_twist = rospy.Subscriber("/cmd_vel", Twist, self.update_message_from_command)
        rospy.loginfo("> Driver Subscriber corrrectly initialized")        
        
        #--- Create the Subscriber to the Image processing module
        self.sub_center = rospy.Subscriber("/lane_detection", Point, self.update_lane)
        rospy.loginfo("Subscribers set")
    
        self.throttle_cmd       = 0.
        self.steer_cmd          = 0.
                  
        self._servo_msg       = ServoArray()
        for i in range(2): self._servo_msg.servos.append(Servo())

        #Mounting model matrixes
        Ac=np.array([[0,1,0,0],[0,-(2*Calfa_front + 2*Calfa_rear)/(mass*Vx),(2*Calfa_front + 2*Calfa_rear)/(mass), -(2*Calfa_front*lf + 2*Calfa_rear*lr)/(mass*Vx)],[0,0,0,1],[0,-(2*Calfa_front*lf - 2*Calfa_rear*lr)/(Iz*Vx),(2*Calfa_front*lf - 2*Calfa_rear*lr)/(Iz),-(2*Calfa_front*lf*lf + 2*Calfa_rear*lr*lr)/(Iz*Vx)]])
        
        Bc=np.array([[0],[(2*Calfa_front)/mass],[0],[(2*Calfa_front)/Iz]])
        
        Ec=np.array([[0],[-(2*Calfa_front*lf - 2*Calfa_rear*lr)/(mass*Vx) - Vx],[0],[-((2*Calfa_front*lf*lf) + (2*Calfa_rear*lr*lr)) / (Iz*Vx)]])
        
        Nstate=len(Ac[0])
        self.Ncontrol=len(Bc[0])
                
        #F = np.array([-Ky, 0, -Kpsi, 0]);
	#Ac_driver = Ac + Bc*F;
	Ac_driver = Ac;
	Bc_driver = Bc;
	Cc_driver = [0,0,0,0];
	Dc_driver = Bc;
	Ad,Bd,Cd,Dd,Tsd = cont2discrete((Ac_driver, Bc_driver, Cc_driver, Dc_driver), Ts, 'zoh')
	Dd = Bd;
        
        #LQR Calculatione
        #self.Klqr=np.array([[0.0300, 0.0061, 0.1140, 0.2627]]) 
        self.Klqr=np.array([[0.0300, 0.0061, 0.055, 0.13]]) 
        
        #Stochastic distribution critical value
        
        #qi_p = 4.2649; #0.001%
        #qi_p = 3.7190; #0.01%
        #qi_p = 3.2905; #0.05%
        #qi_p = 3.0902; #0.1%
        #qi_p = 2.5758; #0.5%
        #qi_p = 2.32634787404084; #1%
        #qi_p = 2.05374891063182; #2%
        qi_p = 1.64485362695147; #5%
        #qi_p = 1.40507156030963; #8%
        #qi_p = 1.28155156554460; #10%
        #qi_p = 1.03643338949379; #15%
        #qi_p = 0.841621233572914; #20%
        
        #Optimization matrices
	PHI = Ad - Bd*self.Klqr;
	
	Hc=np.hstack((Bd,np.zeros((Nstate,N-1))));
	Hw=np.hstack((Dd,np.zeros((Nstate,N-1))));
	
        for i in range(2,N+1):
        	for j in range(1,N+1):
        		if j == 1:
        			Hc_temp = np.matmul(matrix_power(PHI, i-j),Bd);
        			Hw_temp = np.matmul(matrix_power(PHI, i-j),Dd);
        		elif j <= i:
        			temp = np.matmul(matrix_power(PHI, i-j),Bd);
        			Hc_temp = np.hstack((Hc_temp,temp));
        			temp = np.matmul(matrix_power(PHI, i-j),Dd);
        			Hw_temp = np.hstack((Hw_temp,temp));
        		else:
        			temp = np.zeros((Nstate,1));
        			Hc_temp = np.hstack((Hc_temp,temp));
        			Hw_temp = np.hstack((Hw_temp,temp));
        			
        	Hc = np.vstack((Hc,Hc_temp));
        	Hw = np.vstack((Hw,Hw_temp));
        
        for i in range(1,N+1):
        	for j in range(1,N+1):
        		if j == 1:
        			E_prob_temp = np.matmul(matrix_power(PHI, i-j),Dd);
        		elif j <= i:
        			temp = np.matmul(matrix_power(PHI, i-j),Dd);
        			E_prob_temp = np.hstack((E_prob_temp,temp));
        		else:
        			temp = np.zeros((Nstate,1));
        			E_prob_temp = np.hstack((E_prob_temp,temp));
        	if i == 1:
        		E_prob = E_prob_temp;
        	else:			
        		E_prob = np.vstack((E_prob,E_prob_temp));
        
        for i in range (1,N+1):
        	if i == 1:
   			self.sigma_constraint = np.sqrt(w_covar*np.matmul(np.matmul(g_const,Hw[i-1:Nstate,:]),np.matmul(np.transpose(Hw[i-1:Nstate,:]),np.transpose(g_const))))*qi_p;
   		else:
   		 	self.sigma_constraint_temp = np.sqrt(w_covar*np.matmul(np.matmul(g_const,Hw[(i-1)*Nstate:i*Nstate,:]),np.matmul(np.transpose(Hw[(i-1)*Nstate:i*Nstate,:]),np.transpose(g_const))))*qi_p;
   		 	self.sigma_constraint = np.vstack((self.sigma_constraint,self.sigma_constraint_temp));

	for i in range(1,N+1):
        	if i == 1:
        		Hz = matrix_power(PHI, i);
    			Kbar = self.Klqr;
    			G_const = g_const;
    		else:
    			Hz = np.vstack((Hz,matrix_power(PHI, i)));
    			Kbar = block_diag(Kbar, self.Klqr);
    			G_const = block_diag(G_const, g_const);
    	
        #Q_j = np.array([[1, 0, 0, 0],[0, 1.8, 0, 0],[0, 0, 2, 0],[0, 0, 0, 3.5]]);
    	Q_j = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]);
	Q_jf = Q_j;
	r_j = 20;
	R_j = r_j*np.eye(N);
	
	for i in range(1,N):
	   Q_j = block_diag(Q_j, Q_jf);
   	self.Hqp = np.matmul(np.matmul(np.transpose(Hc),Q_j),Hc) + np.matmul(np.matmul(np.transpose(np.matmul(Kbar,Hc) + np.identity(N*self.Ncontrol)),R_j),(np.matmul(Kbar,Hc) + np.identity(N*self.Ncontrol)));
  
   	Fqp_1 = np.matmul(np.matmul(np.transpose(Hz),np.transpose(Kbar)),np.matmul(np.transpose(Hc),np.transpose(Kbar)))+np.matmul(np.transpose(Hz),np.transpose(Kbar))+np.matmul(np.transpose(Hz),Hc);
   	Fqp_2 = np.matmul(np.matmul(np.transpose(Hw),np.transpose(Kbar)),np.matmul(np.transpose(Hc),np.transpose(Kbar)))+np.matmul(np.transpose(Hw),np.transpose(Kbar))+np.matmul(np.transpose(Hw),Hc);
   	self.Fqp = np.vstack((Fqp_1,Fqp_2));
   	A_cmax = np.matmul(Kbar,Hc) + np.identity(N*self.Ncontrol);
	b_cmax = vmax*np.ones((N*self.Ncontrol,1));
	A_cmin = np.matmul(-Kbar,Hc) - np.identity(N*self.Ncontrol);
	b_cmin = -vmin*np.ones((N*self.Ncontrol,1));
   
	A_epislon_max = np.matmul(G_const,Hc);
	b_epislon_max = epsilon_max_limits*np.ones((N, 1));
	A_epislon_min = -np.matmul(G_const,Hc);
	b_epislon_min = -epsilon_min_limits*np.ones((N, 1));

	Aineq = np.vstack((A_cmax,A_cmin,A_epislon_max,A_epislon_min));
	bineq = np.vstack((b_cmax,b_cmin,b_epislon_max,b_epislon_min));
	
	
	# Solver Setup
	#options = optimoptions('quadprog','Display','None');
	# Create an OSQP object
	self.prob_smpc_8 = osqp.OSQP();

	Hqp_csc = sparse.csc_matrix(self.Hqp);
	Aineq = sparse.csc_matrix(Aineq);

	# Setup workspace
	self.prob_smpc_8.setup(Hqp_csc, np.zeros((N)), Aineq, -np.inf*np.ones((N*Nstate,1)), bineq, warm_start=False, verbose=False);

	# Safety Constraints
	self.epsilon_max_array = epsilon_max_limits*np.ones((N + Nsim, 1));
	self.epsilon_min_array = epsilon_min_limits*np.ones((N + Nsim, 1));

	#Referencia negativa
	time_constraint = (dist_constraint1+(Ts*Vx))/Vx;
	n_constraint1_ini = np.round(time_constraint/Ts);
	n_constraint1_end = np.round(n_constraint1_ini+distlength/(Ts*Vx));
	n_constraint1_length = np.round(n_constraint1_end - n_constraint1_ini);
	n_constraint1_ini = n_constraint1_ini.astype(int);
	n_constraint1_end = n_constraint1_end.astype(int);
	n_constraint1_length_int = n_constraint1_length.astype(int);
	
	time_constraint = (dist_constraint2+(Ts*Vx))/Vx;
	n_constraint2_ini = np.round(time_constraint/Ts);
	n_constraint2_end = np.round(n_constraint2_ini+distlength/(Ts*Vx));
	n_constraint2_length = np.round(n_constraint2_end - n_constraint2_ini);
	n_constraint2_ini = n_constraint2_ini.astype(int);
	n_constraint2_end = n_constraint2_end.astype(int);
	n_constraint2_length_int = n_constraint2_length.astype(int);
	
	#for i in range(1,n_constraint1_length_int):
	#	self.epsilon_min_array[n_constraint1_ini+i] = self.epsilon_min_array[n_constraint1_ini+i] -epsilon_min_limits*(i/n_constraint1_length);
	
	#for i in range(0,n_constraint2_length_int):
	#	self.epsilon_min_array[n_constraint2_ini+i] = -epsilon_min_limits*(-((i)/n_constraint2_length));
		
	#for i in range(n_constraint1_end,n_constraint2_ini):
	##for i in range(n_constraint1_end,N + Nsim):
	#	self.epsilon_min_array[i] = 0;
		
	for i in range(1,n_constraint1_length_int):
		self.epsilon_min_array[n_constraint1_ini+i] = self.epsilon_min_array[n_constraint1_ini+i] -(epsilon_min_limits-0.1)*(i/n_constraint1_length);
	
	for i in range(0,n_constraint2_length_int+2):
		self.epsilon_min_array[n_constraint2_ini+i] = 0.1-(epsilon_min_limits)*(-((i)/n_constraint2_length));
		
	for i in range(n_constraint1_end,n_constraint2_ini):
	##for i in range(n_constraint1_end,N + Nsim):
		self.epsilon_min_array[i] = 0.1;
	
	#Referencia positiva
	time_constraint = (dist_constraint3+(Ts*Vx))/Vx;
	n_constraint3_ini = np.round(time_constraint/Ts);
	n_constraint3_end = np.round(n_constraint3_ini+distlength/(Ts*Vx));
	n_constraint3_length = np.round(n_constraint3_end - n_constraint3_ini);
	n_constraint3_ini = n_constraint3_ini.astype(int);
	n_constraint3_end = n_constraint3_end.astype(int);
	n_constraint3_length_int = n_constraint3_length.astype(int);
	
	time_constraint = (dist_constraint4+(Ts*Vx))/Vx;
	n_constraint4_ini = np.round(time_constraint/Ts);
	n_constraint4_end = np.round(n_constraint4_ini+distlength/(Ts*Vx));
	n_constraint4_length = np.round(n_constraint4_end - n_constraint4_ini);
	n_constraint4_ini = n_constraint4_ini.astype(int);
	n_constraint4_end = n_constraint4_end.astype(int);
	n_constraint4_length_int = n_constraint4_length.astype(int);
	
	for i in range(1,n_constraint3_length_int):
		self.epsilon_max_array[n_constraint3_ini+i] = self.epsilon_max_array[n_constraint3_ini+i] -epsilon_max_limits*(i/n_constraint3_length);
	
	for i in range(0,n_constraint4_length_int):
		self.epsilon_max_array[n_constraint4_ini+i] = epsilon_max_limits*(((i)/n_constraint4_length));
	
		
	for i in range(n_constraint3_end,n_constraint4_ini):
	##for i in range(n_constraint3_end,N + Nsim):
		self.epsilon_max_array[i] = 0;
		
	#for i in range(1,n_constraint3_length_int-1):
	#	self.epsilon_max_array[n_constraint3_ini+i] = self.epsilon_max_array[n_constraint3_ini+i] -(epsilon_max_limits)*(i/n_constraint3_length);
	
	#for i in range(0,n_constraint4_length_int-1):
	#	self.epsilon_max_array[n_constraint4_ini+i] = 0.1+epsilon_max_limits*(((i)/n_constraint4_length));
	
		
	#for i in range(n_constraint3_end-1,n_constraint4_ini):
	##for i in range(n_constraint3_end,N + Nsim):
	#	self.epsilon_max_array[i] = 0.1;
	
	
	
	#Restricao 3
	time_constraint = (dist_constraint5+(Ts*Vx))/Vx;
	n_constraint5_ini = np.round(time_constraint/Ts);
	n_constraint5_end = np.round(n_constraint5_ini+distlength/(Ts*Vx));
	n_constraint5_length = np.round(n_constraint5_end - n_constraint5_ini);
	n_constraint5_ini = n_constraint5_ini.astype(int);
	n_constraint5_end = n_constraint5_end.astype(int);
	n_constraint5_length_int = n_constraint5_length.astype(int);
	
	time_constraint = (dist_constraint6+(Ts*Vx))/Vx;
	n_constraint6_ini = np.round(time_constraint/Ts);
	n_constraint6_end = np.round(n_constraint6_ini+distlength/(Ts*Vx));
	n_constraint6_length = np.round(n_constraint6_end - n_constraint6_ini);
	n_constraint6_ini = n_constraint6_ini.astype(int);
	n_constraint6_end = n_constraint6_end.astype(int);
	n_constraint6_length_int = n_constraint6_length.astype(int);
	
	#for i in range(1,n_constraint5_length_int):
	#	self.epsilon_min_array[n_constraint5_ini+i] = self.epsilon_min_array[n_constraint5_ini+i] -epsilon_min_limits*(i/n_constraint5_length);
	
	#for i in range(0,n_constraint6_length_int):
	#	self.epsilon_min_array[n_constraint6_ini+i] = -epsilon_min_limits*(-((i)/n_constraint6_length));
		
	#for i in range(n_constraint5_end,n_constraint6_ini):
	##for i in range(n_constraint1_end,N + Nsim):
	#	self.epsilon_min_array[i] = 0;
	
	for i in range(1,n_constraint5_length_int):
		self.epsilon_min_array[n_constraint5_ini+i] = self.epsilon_min_array[n_constraint5_ini+i] -(epsilon_min_limits-0.1)*(i/n_constraint5_length);
	
	for i in range(0,n_constraint6_length_int+2):
		self.epsilon_min_array[n_constraint6_ini+i] = 0.1-(epsilon_min_limits)*(-((i)/n_constraint6_length));
		
	for i in range(n_constraint5_end,n_constraint6_ini):
	##for i in range(n_constraint1_end,N + Nsim):
		self.epsilon_min_array[i] = 0.1;

	

	self.W_array = w_bar*np.ones((N*self.Ncontrol,1));
	
	self.KH = np.hstack((np.matmul(-Kbar,Hz),np.matmul(Kbar,Hw)))
	self.GH = np.hstack((np.matmul(G_const,Hz),np.matmul(G_const,Hw)))
        
        
        self.offset         = 0.0
        self.angle         = 0.0
        self._time_detected = 0.0
        
       

        self._time_steer        = 0
        self._steer_sign_prev   = 0
        
        #self.Speed_PWM =(V_pwm+75.0270218458018)/0.216820297418678
        

        
    @property
    
    def is_detected(self): return(time.time() - self._time_detected < 1.0)
        
    def update_lane(self, message):
        self.offset = message.x
        self.angle = message.y
        self._time_detected = time.time()

    def update_message_from_command(self, message):
        self._last_time_cmd_rcv = time.time()
        self.throttle_cmd       = message.linear.x
        self.steer_cmd          = message.angular.z

 
    def run(self):
        
        #--- Set the control rate
        rate = rospy.Rate(1/Ts)
        x2=0;
        x4=0;
        cont_cant_u_smpc_8 = 0;
        
        cont_ini=0
        cont=0
           
        #Routine to initialize the motors
        while cont_ini<=2/Ts:
        
            self._servo_msg.servos[0].servo = 1
            self._servo_msg.servos[0].value = self.Speed_PWM
            self._servo_msg.servos[1].servo = 2
            self._servo_msg.servos[1].value = PWM_Angle_Center
            self.ros_pub_servo_array.publish(self._servo_msg)
            rospy.loginfo("Speed Ramp up %d", cont_ini)
            x1 = self.offset
            x1_k_1 = x1
            x1_k_2 = x1
            x1_avg_old = x1
            x3 = self.angle
            x3_k_1 = x3
            x3_k_2 = x3
            x3_avg_old = x3

            cont_ini=cont_ini+1
            rate.sleep()
            
        #Control Routine   
        while cont_ini > 2/Ts and cont<=Nsim:
        
            #Moving average filter        		
            x1_k_2 = x1_k_1
            x1_k_1 = x1
            x1 = self.offset
            x1_avg = (x1_k_2 + x1_k_1 + x1)/3
            
            x3_k_2 = x3_k_1
            x3_k_1 = x3
            x3 = self.angle
            x3_avg = (x3_k_2 + x3_k_1 + x3)/3
            
            #States
            x_smpc_8 = np.array([[x1_avg],[x1_avg-x1_avg_old],[x3_avg],[x3_avg-x3_avg_old]]);
            x2=x1_avg_old-x1_avg;
            x4=x3_avg_old-x3_avg;
            x1_avg_old = x1_avg
            x3_avg_old = x3_avg
            
            #Safety restrictions
            Epsilon_max = self.epsilon_max_array[0+cont:(N+cont),:];
            Epsilon_min = self.epsilon_min_array[0+cont:(N+cont),:];
            
            b_cmax = vmax*np.ones((N*self.Ncontrol,1)) - np.matmul(self.KH,np.vstack((x_smpc_8, self.W_array)));
            b_cmin = -vmin*np.ones((N*self.Ncontrol,1)) + np.matmul(self.KH,np.vstack((x_smpc_8, self.W_array)));

            b_epislon_max = Epsilon_max - np.matmul(self.GH,np.vstack((x_smpc_8, self.W_array))) - self.sigma_constraint;
            b_epislon_min = -Epsilon_min + np.matmul(self.GH,np.vstack((x_smpc_8, self.W_array))) - self.sigma_constraint;
            
            bineq = np.vstack((b_cmax,b_cmin,b_epislon_max,b_epislon_min));
            
            self.prob_smpc_8.update(q=np.reshape(np.matmul(np.transpose(np.vstack((x_smpc_8,self.W_array))),self.Fqp),(N,)), u=bineq);
            res_smpc_8 = self.prob_smpc_8.solve();
                     
            # Check solver status
    	    if res_smpc_8.info.status == 'Solved':
            	c_smpc_8 = res_smpc_8.x[-N];
            else:
            	print('OSQP did not solve the problem!');
            	cont_cant_u_smpc_8 = cont_cant_u_smpc_8 + 1;
            	c_smpc_8 = c_smpc_8_old;
            c_smpc_8_old = c_smpc_8;
            
            v_smpc_8 = np.matmul(-self.Klqr,x_smpc_8) + c_smpc_8;
            
            #Conversion from angle to PWM
            if v_smpc_8 >= 0:
            	angle_PWM =-34592.57672675*((v_smpc_8))**4+22479.5266456373*((v_smpc_8))**3-4625.27066341302*((v_smpc_8))**2+504.425631251666*((v_smpc_8)) + PWM_Angle_Center
            else:
            	angle_PWM =44698.3289950114*((v_smpc_8))**4+24920.6338823119*((v_smpc_8))**3+4569.41008015062*((v_smpc_8))**2+518.538021665201*((v_smpc_8)) + PWM_Angle_Center
            	
            #Publish control actions
            self._servo_msg.servos[0].servo = 1
            self._servo_msg.servos[0].value = round(self.Speed_PWM)
            self._servo_msg.servos[1].servo = 2
            self._servo_msg.servos[1].value = round(angle_PWM)
            self.ros_pub_servo_array.publish(self._servo_msg)
            
            #Publish safety restriction debug topic
            self.safe_restriction.x = self.epsilon_max_array[cont]
	    self.safe_restriction.y = self.epsilon_min_array[cont]
	    self.safe_pub.publish(self.safe_restriction) 
	    
	    cont=cont+1
            
            #User visualization
            rospy.loginfo("%d , %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f",angle_PWM, v_smpc_8, c_smpc_8, x1_avg, x2, x3_avg, x4, self.safe_restriction.x, self.safe_restriction.y)
            
	    rate.sleep()
	
	#Stop the car after tests
	self._servo_msg.servos[0].servo = 1
	self._servo_msg.servos[0].value = 333
	self._servo_msg.servos[1].servo = 2
	self._servo_msg.servos[1].value = 333
	self.ros_pub_servo_array.publish(self._servo_msg)
                   
            
if __name__ == "__main__":

    rospy.init_node('control_node')
    
    control_node     = ControlNode()
    control_node.run()            
