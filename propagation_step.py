import random
import math
import numpy as np
import scipy
from scipy.linalg import expm


class particlefilter():


    def __init__(self, dt, num_of_particles, lw_vel, rw_vel, r, w, std_l, std_r, std_p):
        self.dt = dt
        self.num_of_particles = num_of_particles
        self.lw_vel = lw_vel
        self.rw_vel = rw_vel
        self.r = r
        self.w = w
        self.std_l = std_l
        self.std_r = std_r
        self.std_p = std_p
        

    def wheel_speeds(self):
        noise_left = np.random.normal(0, self.std_l)
        noise_right = np.random.normal(0, self.std_r)
        left_speed = self.lw_vel + noise_left
        right_speed = self.rw_vel + noise_right

        return left_speed, right_speed


    def lie_element(self, left_speed, right_speed):
        lie_element = np.array([[0, (-self.r/self.w)*(right_speed - left_speed), (self.r/2)*(right_speed+left_speed)], [(self.r/self.w)*(right_speed - left_speed), 0, 0], [0,0,0] ])
        return lie_element
    
    def rand_positions(self):
        self.x = np.round(np.random.normal(loc=0, scale=1, size=self.num_of_particles),2)
        self.y = np.round(np.random.normal(loc=0, scale=1, size=self.num_of_particles),2)
        self.theta_angles = [0, 3.14, 1.57, 0.785]


    def identity_particles(n):
        return [np.eye(3) for _ in range(n)]

    def propagation_step(self, particles):
        x_t_plus_set = []
        for i in range(particles):

            left_speed = self.wheel_speeds()[0]
            right_speed = self.wheel_speeds()[1]

            lie_element = self.lie_element(left_speed, right_speed)
            theta = np.random.choice(self.theta_angles)  
            x_t = np.array([[math.cos(theta), -math.sin(theta), x[i]], [math.sin(theta), math.cos(theta), y[i]], [0, 0, 1]])

            x_t_plus= x_t @ expm(self.dt*lie_element)
            x_t_plus_set.append(x_t_plus)
        return x_t_plus_set     #this is the set of transformations of all the particles


    def update(self, particles):
        
        z = (1.3, 1.5)
        l_t = self.propagation_step()

        x_y = []
        for particle in l_t:
            x = particle[0,2]
            y = particle[1,2]
            x_y.append((x,y))
        error = z - l_t
        exp = -0.5*((error)**2)/(self.std_p)**2
        weight = 0.5*math.pi*self.std_p*expm(exp)



        z = l_t + noise
        weights_x = []
        weights_y = []
        for x_t in particles:
            weight_x = z/x_t[0,2]
            weight_y = z/x_t[1,2]
            weights_x.append(weight_x)
            weights_y.append(weight_y)
        x_weight_sum = sum(weights_x)
        y_weight_sum = sum(weights_y)
        normalized_x = []
        normalized_y = []
        for i in weights_x:
            normalize_x = i / x_weight_sum
            normalize_y = i / y_weight_sum
            normalized_x.append(normalize_x)
            normalized_y.append(normalize_y)
        for i in normalized_x:
            
        meas_noise = np.random.normal(0, (noise_mag)**2)
        noise = 0.5*(noise_mag)**2 * (expm(0.5*))

        z = l_t + noise

        x_updated = 

        return updated_x



if __name__ == "__main__":
        dt = 5
        num_of_particles = 1
        lw_vel = 1.5
        rw_vel = 2
        r = 0.25
        w = 0.5
        std_l = 0.05
        std_r = 0.10
        std_p = 0.20
    
pf = particlefilter(dt, num_of_particles, lw_vel, rw_vel, r, w, std_l, std_r)

print(pf.propagation_step(num_of_particles))



#print(propagation_step(1, 1000, 1.5, 2, 10, 0.25, 0.5, 0.05, 0.05))

















