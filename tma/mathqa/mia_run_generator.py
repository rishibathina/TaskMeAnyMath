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