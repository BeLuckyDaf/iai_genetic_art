import random
from numba import jit
import numpy as np
import cv2


@jit(nopython=True)
def mutate_matrix(image, rate, offset, block_size):
    image_length_y = len(image)
    image_length_x = len(image[0])
    for y in range(0, image_length_y, block_size):
        for x in range(0, image_length_x, block_size):
            mutated = random.uniform(0, 1) < rate
            if mutated:
                r = random.randint(-offset, offset)
                g = random.randint(-offset, offset)
                b = random.randint(-offset, offset)
                for cy in range(y, y+block_size):
                    for cx in range(x, x+block_size):
                        image[cy, cx, 0] = min(max(image[cy, cx, 0]+b, 0), 255)
                        image[cy, cx, 1] = min(max(image[cy, cx, 1]+g, 0), 255)
                        image[cy, cx, 2] = min(max(image[cy, cx, 2]+r, 0), 255)


@jit(nopython=True)
def mutate_matrix_random(image, rate, offset, block_size):
    image_length_y = len(image)
    image_length_x = len(image[0])
    count = (image_length_x*image_length_y) // (block_size**2)
    for i in range(count):
        mutated = random.uniform(0, 1) < rate
        if mutated:
            r = random.randint(-offset, offset)
            g = random.randint(-offset, offset)
            b = random.randint(-offset, offset)
            y = random.randint(0, image_length_y)
            x = random.randint(0, image_length_x)
            for cy in range(y, min(y+block_size, image_length_y)):
                for cx in range(x, min(x+block_size, image_length_x)):
                    image[cy, cx, 0] = min(max(image[cy, cx, 0]+b, 0), 255)
                    image[cy, cx, 1] = min(max(image[cy, cx, 1]+g, 0), 255)
                    image[cy, cx, 2] = min(max(image[cy, cx, 2]+r, 0), 255)


@jit(nopython=True)
def mutate_and_add(image, times, rate, offset, block_size):
    result = [image]
    for i in range(times):
        current_image = image.copy()
        mutate_matrix(current_image, rate, offset, block_size)
        result.append(current_image)
    return result


@jit(nopython=True)
def crossover(img1, img2, block_size):
    result = img1.copy()
    image_length_y = len(img1)
    image_length_x = len(img1[0])
    count = (image_length_x*image_length_y) // (block_size**2)
    for i in range(count):
        kind = random.randint(0, 7)
        y = random.randint(0, image_length_y)
        x = random.randint(0, image_length_x)
        for cy in range(max(y-(block_size//2), 0), min(y+(block_size//2), image_length_y)):
            for cx in range(max(x-(block_size//2), 0), min(x+(block_size//2), image_length_x)):
                if kind == 0:
                    result[cy, cx, 0] = img2[cy, cx, 0]
                    result[cy, cx, 1] = img2[cy, cx, 1]
                    result[cy, cx, 2] = img2[cy, cx, 2]
                elif kind == 2:
                    result[cy, cx, 0] = img2[cy, cx, 0]
                elif kind == 3:
                    result[cy, cx, 1] = img2[cy, cx, 1]
                elif kind == 4:
                    result[cy, cx, 2] = img2[cy, cx, 2]
                elif kind == 5:
                    result[cy, cx, 1] = img2[cy, cx, 1]
                    result[cy, cx, 2] = img2[cy, cx, 2]
                elif kind == 6:
                    result[cy, cx, 0] = img2[cy, cx, 0]
                    result[cy, cx, 2] = img2[cy, cx, 2]
                elif kind == 7:
                    result[cy, cx, 0] = img2[cy, cx, 0]
                    result[cy, cx, 1] = img2[cy, cx, 1]
    return result


@jit(nopython=True)
def fitness(image, target, goal):
    image_length_x = len(image[0])
    image_length_y = len(image)
    passed = 0
    total = image_length_x * image_length_y
    for y in range(0, image_length_y):
        for x in range(0, image_length_x):
            points = 0
            points += max(image[y, x, 0], target[y, x, 0]) - min(image[y, x, 0], target[y, x, 0])
            points += max(image[y, x, 1], target[y, x, 1]) - min(image[y, x, 1], target[y, x, 1])
            points += max(image[y, x, 2], target[y, x, 2]) - min(image[y, x, 2], target[y, x, 2])
            points /= 3
            if points > 0:
                passed += min(goal/points, 1)
    return passed/total


@jit(nopython=True)
def crossover_2x2(img1, img2):
    result = np.zeros((2,2,3))
    for y in range(2):
        for x in range(2):
            seed = random.randint(0, 7)
            if seed == 0:
                result[y, x, 0] = img1[y, x, 0]
                result[y, x, 1] = img1[y, x, 1]
                result[y, x, 2] = img1[y, x, 2]
            elif seed == 1:
                result[y, x, 0] = img2[y, x, 0]
                result[y, x, 1] = img2[y, x, 1]
                result[y, x, 2] = img2[y, x, 2]
            elif seed == 2:
                result[y, x, 0] = img1[y, x, 0]
                result[y, x, 1] = img2[y, x, 1]
                result[y, x, 2] = img2[y, x, 2]
            elif seed == 3:
                result[y, x, 0] = img2[y, x, 0]
                result[y, x, 1] = img1[y, x, 1]
                result[y, x, 2] = img2[y, x, 2]
            elif seed == 4:
                result[y, x, 0] = img2[y, x, 0]
                result[y, x, 1] = img2[y, x, 1]
                result[y, x, 2] = img1[y, x, 2]
            elif seed == 5:
                result[y, x, 0] = img2[y, x, 0]
                result[y, x, 1] = img1[y, x, 1]
                result[y, x, 2] = img1[y, x, 2]
            elif seed == 6:
                result[y, x, 0] = img1[y, x, 0]
                result[y, x, 1] = img2[y, x, 1]
                result[y, x, 2] = img1[y, x, 2]
            elif seed == 7:
                result[y, x, 0] = img1[y, x, 0]
                result[y, x, 1] = img1[y, x, 1]
                result[y, x, 2] = img2[y, x, 2]
    return result


@jit(nopython=True)
def mutate_matrix_2x2(image, rate, offset):
    mutate_matrix(image, rate, offset, 1)


def selection(population, fitlist, count):
    new_population = []
    for i in range(count):
        if i >= len(fitlist):
            break
        max_fitness_index = 0
        max_fitness = 0
        for i in range(len(fitlist)):
            if fitlist[i] > max_fitness:
                max_fitness_index = i
                max_fitness = fitlist[i]
        new_population.append(population[max_fitness_index].copy())
        fitlist[max_fitness_index] = -1
    return new_population


def get_fitlist(population, target, goal):
    result = []
    for p in population:
        fit = fitness(p, target, goal)
        result.append(fit)
    return result


def calculate(image, target, epochs=50, goal=10, mutation_rate=0.5, offset=50, select=5, initial_pop=30):
    population = [image]
    for i in range(initial_pop):
        mut = image.copy()
        mutate_matrix_2x2(mut, 1, offset)
        population.append(mut)
    
    fitlist = get_fitlist(population, target, goal)
    new_population = selection(population, fitlist, select)
    children = []
    for iteration in range(epochs):
        population = new_population
        for i in range(len(new_population)-1):
            for j in range(i+1, len(new_population)):
                child = crossover_2x2(new_population[i], new_population[j])
                mutate_matrix_2x2(child, mutation_rate, offset)
                children.append(child)
        new_population += children
        fitlist = get_fitlist(population, target, goal)
        new_population = selection(population, fitlist, select)
    fitlist = get_fitlist(new_population, target, goal)
    return new_population[0]


def calculate_image(image, target, epochs=10, goal=10, mutation_rate=0.5, offset=50, select=5, initial_pop=30):
    image_copy = image.copy()
    image_length_y = len(image_copy)
    image_length_x = len(image_copy[0])
    for y in range(0, image_length_y, 2):
        for x in range(0, image_length_x, 2):
            point = calculate(image_copy[y:y+2, x:x+2], target[y:y+2, x:x+2], epochs, goal, mutation_rate, offset, select, initial_pop)
            image_copy[y, x] = point[0, 0]
            image_copy[y, x+1] = point[0, 1]
            image_copy[y+1, x] = point[1, 0]
            image_copy[y+1, x+1] = point[1, 1]
        print("Genetic: {:.1f}%".format(100*y/image_length_y))
    print("Genetic: 100.0%")
    return image_copy


def get_average_color(image):
    image_length_y = len(image)
    image_length_x = len(image[0])
    r = 0
    g = 0
    b = 0
    for y in range(0, image_length_y):
        for x in range(0, image_length_x):
            b += image[y, x, 0]
            g += image[y, x, 1]
            r += image[y, x, 2]
    pixels = image_length_x * image_length_y
    return [b//pixels, g//pixels, r//pixels]


def calculate_average_image(image, block=4):
    image_copy = image.copy()
    image_length_y = len(image_copy)
    image_length_x = len(image_copy[0])
    for y in range(0, image_length_y, block):
        for x in range(0, image_length_x, block):
            average = get_average_color(image_copy[y:y+block, x:x+block])
            for by in range(y, y+block):
                for bx in range(x, x+block):
                    image_copy[by,bx] = average.copy()
        print("Pixelize: {:.1f}%".format(100*y/image_length_y))
    print("Pixelize: 100.0%")
    return image_copy


@jit(nopython=True)
def generate_random_image():
    img = np.zeros((512,512,3), np.uint8)
    for y in range(512):
        for x in range(512):
            img[y,x,0] = random.randint(0,255)
            img[y,x,1] = random.randint(0,255)
            img[y,x,2] = random.randint(0,255)
    return img
