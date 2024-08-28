template_path = 'annotations/math_annotations/angle_sum_templates.json'
metadata = MathTemplateMetaData(template_path=template_path)
generator = AngleSumGenerator(metadata)
task_store = TaskStore(AngleSumGenerator.schema)

generator.enumerate_task_plans(task_store)
all_task_plans = task_store.return_df().to_dict(orient='records')

generated_tasks = set()
num_task = 0

def get_hashable_task_plan(task_plan):
    return json.dumps(task_plan, sort_keys=True)

while num_task < 50 and all_task_plans:
    random_task_plan = random.choice(all_task_plans)
    hashable_task_plan = get_hashable_task_plan(random_task_plan)
    if hashable_task_plan not in generated_tasks:
        generated_tasks.add(hashable_task_plan)
        task = generator._generate_task(random_task_plan)
        question, answer, _ = task
        print(question)
        print("Answer: " + str(answer))
        num_task += 1
        print("Task count: ", num_task)
    all_task_plans.remove(random_task_plan)

if num_task < 50:
    print("Less than 50 unique tasks available. Generated:", num_task)
    
    
# run sphere generator

template_path = 'annotations/math_annotations/sphere_volume_templates.json'
metadata = MathTemplateMetaData(template_path)
generator = VolumeSphereGenerator(metadata)
task_store = TaskStore(VolumeSphereGenerator.schema)
generator.enumerate_task_plans(task_store)
all_tasks = task_store.return_df().to_dict(orient='records')
for i in range(0, 50):
    random_task_plan = random.choice(all_tasks)
    task = generator._generate_task(random_task_plan)
    question, answer, _ = task
    print(question)
    print("Answer:", answer)
    
# run cone volume generator
template_path = 'annotations/math_annotations/cone_volume_templates.json'
metadata = MathTemplateMetaData(template_path)
generator = ConeVolumeGenerator(metadata)
task_store = TaskStore(ConeVolumeGenerator.schema)
generator.enumerate_task_plans(task_store)
all_tasks = task_store.return_df().to_dict(orient='records')

for i in range(0, 50):
    random_task_plan = random.choice(all_tasks)
    task = generator._generate_task(random_task_plan)
    question, answer, _ = task
    print(question)
    print("Answer:", answer)
    
# run the triangle area class
template_path = 'annotations/math_annotations/triangle_area_templates.json'
metadata = MathTemplateMetaData(template_path=template_path)
generator = TriangleAreaGenerator(metadata, multiple_choice = True)
task_store = TaskStore(TriangleAreaGenerator.schema)
generator.enumerate_task_plans(task_store)
all_tasks = task_store.return_df().to_dict(orient='records')

for i in range(0, 50):
    random_task_plan = random.choice(all_tasks)
    task = generator._generate_task(random_task_plan)
    question, answer, options, _ = task
    print(question)
    print("Answer:", answer)
    print("option1: "+ str(options[0]) + " and option2: " + str(options[1]) + " and option2: " + str(options[2]))
    
# run the VolumeRectangularPrismGenerator
template_path = 'annotations/math_annotations/volume_rectangular_prism_templates.json'
metadata = MathTemplateMetaData(template_path=template_path)
generator = VolumeRectangularPrismGenerator(metadata, multi_options = True)
task_store = TaskStore(VolumeRectangularPrismGenerator.schema)
generator.enumerate_task_plans(task_store)
all_tasks = task_store.return_df().to_dict(orient='records')

for i in range(0, 50):
    random_task_plan = random.choice(all_tasks)
    task = generator._generate_task(random_task_plan)
    question, answer, options, _ = task
    print(question)
    print("Answer:", answer)
    print("option1: "+ str(options[0]) + " and option2: " + str(options[1]) + " and option2: " + str(options[2]))
    
# run angle generator
template_path = 'annotations/math_annotations/angle_template.json'
metadata = MathTemplateMetaData(template_path)
generator = AngleGenerator(metadata)
task_store = TaskStore(AngleGenerator.schema)
generator.enumerate_task_plans(task_store)
all_tasks = task_store.return_df().to_dict(orient='records')
for i in range(0, 50):
    random_task_plan = random.choice(all_tasks)
    task = generator._generate_task(random_task_plan)
    question, answer, _ = task
    print(question)
    print("Answer:", answer)
    
# run the matrix
template_path = 'annotations/math_annotations/matrix_templates.json'
metadata = MathTemplateMetaData(template_path)
generator = MatrixProblemGenerator(metadata, multiple_choice = True)
task_store = TaskStore(MatrixProblemGenerator.schema)
generator.enumerate_task_plans(task_store)
all_tasks = task_store.return_df().to_dict(orient='records')
for i in range(0, 50):
    random_task_plan = random.choice(all_tasks)
    task = generator._generate_task(random_task_plan)
    question, options, answer, _ = task
    print(question)
    print("Answer: ", options)
    
# the linear_systems problems
template_path = 'annotations/math_annotations/linear_system_templates.json'
metadata = MathTemplateMetaData(template_path)
generator = LinearEquationSystemGenerator(metadata, num_equations=2, multiple_choice=True)
task_store = TaskStore(LinearEquationSystemGenerator.schema)

generator.enumerate_task_plans(task_store)
all_tasks = task_store.return_df().to_dict(orient='records')

for i in range(0, 50):
    random_task_plan = random.choice(all_tasks)
    task = generator._generate_task(random_task_plan)
    question, options, answer, _ = task
    print(question)
    print("Answer:", answer)
    if options:
        for key, value in options.items():
            print(f"{key}: {value}")
            
# the basic operations problems
template_path = 'annotations/math_annotations/basic_operation_templates.json'
metadata = MathTemplateMetaData(template_path)
generator = BasicArithmeticOperationsGenerator(metadata, multiple_choice = True)
task_store = TaskStore(BasicArithmeticOperationsGenerator.schema)
generator.enumerate_task_plans(task_store)
all_tasks = task_store.return_df().to_dict(orient='records')
for i in range(0, 50):
    random_task_plan = random.choice(all_tasks)
    task = generator._generate_task(random_task_plan)
    question, options, answer, _ = task
    print(question)
    print("Answer: ", options)
    
# simple exponents
template_path = 'annotations/math_annotations/exponent_templates.json'
metadata = MathTemplateMetaData(template_path)
generator = SimpleExponentGenerator(metadata, multiple_choice = True)
task_store = TaskStore(SimpleExponentGenerator.schema)
generator.enumerate_task_plans(task_store)
all_tasks = task_store.return_df().to_dict(orient='records')
for i in range(0, 50):
    random_task_plan = random.choice(all_tasks)
    task = generator._generate_task(random_task_plan)
    question, options, answer, _ = task
    print()
    print(question)
    print("Answer: ", options)