import sys, os
import numpy as np

def get_adaptation_stats(stats_dict, max_frames, start_frame=0):
    num_updates = 0
    num_samples = 0
    for f in sorted(stats_dict.keys()):
        if f > max_frames:
            break
        if f < start_frame:
            continue
        num_samples = num_samples + stats_dict[f]['ran_teacher']
        if stats_dict[f]['num_updates'] is not None:
            num_updates = num_updates + stats_dict[f]['num_updates']

    return num_samples, num_updates

def total_cost_jitnet(total_frames, num_samples, num_updates,
               teacher_inference_cost, jitnet_inference_cost,
               jitnet_train_cost):
    return (total_frames - num_samples) * jitnet_inference_cost + \
            (teacher_inference_cost * num_samples) + \
            (num_updates * jitnet_train_cost)

def total_cost_flow(total_frames, num_samples,
                    teacher_inference_cost, flow_cost):
    return (total_frames - num_samples) * flow_cost + \
            (teacher_inference_cost * num_samples)

def compute_per_class_stats(stats_dict, max_frames, num_classes,
                            start_frame = 0):
    segment_stats = { 'tp': np.zeros(num_classes, np.float32),
                      'fp': np.zeros(num_classes, np.float32),
                      'fn': np.zeros(num_classes, np.float32),
                    }
    for f in sorted(stats_dict.keys()):
        if f > max_frames:
            break
        if f < start_frame:
            continue
        for cls in range(num_classes):
            segment_stats['tp'][cls] = segment_stats['tp'][cls] + stats_dict[f]['tp'][cls]
            segment_stats['fp'][cls] = segment_stats['fp'][cls] + stats_dict[f]['fp'][cls]
            segment_stats['fn'][cls] = segment_stats['fn'][cls] + stats_dict[f]['fn'][cls]

    eps = 1e-06
    cls_ious = []
    cls_fraction = []
    total_pixel = 0
    for cls in range(1, num_classes):
        total_pixel += segment_stats['tp'][cls] + segment_stats['fn'][cls]
    for cls in range(num_classes):
        iou = (segment_stats['tp'][cls] + eps)/(segment_stats['tp'][cls] + segment_stats['fn'][cls] + segment_stats['fp'][cls] + eps)
        cls_ious.append(iou)
        if cls == 0:
            cls_fraction.append(0)
        else:
            cls_fraction.append((segment_stats['tp'][cls] + segment_stats['fn'][cls]) / (total_pixel + eps))

    return cls_ious, cls_fraction

def make_table(class_names, stats_variants, max_frames, file_name, exclude_classes,
               start_frame = 0):
    csv_file = open(file_name, 'w')
    csv_file.write(",")
    csv_file.write(",".join([cls_name for cls_name in class_names if class_names not in exclude_classes] + ['Speed up'] + ['Samples'] + ['Updates']))
    csv_file.write('\n')
    for s, name in stats_variants:
        stats_dict = np.load(s, allow_pickle=True)[0]
        num_classes = len(class_names)
        cls_ious, fractions = compute_per_class_stats(stats_dict, max_frames,
                                                      num_classes,
                                                      start_frame=start_frame)
        num_samples, num_updates = get_adaptation_stats(stats_dict, max_frames,
                                                        start_frame=start_frame)
        total_frames = min(len(stats_dict), max_frames - start_frame)
        online_cost = total_cost_jitnet(total_frames, num_samples, num_updates,
                                        300, 7, 30)
        teacher_cost = total_cost_jitnet(total_frames, total_frames, num_updates,
                                  300, 0, 0)
        speedup = float(teacher_cost)/online_cost

        cls_ious = [ 100 * iou for idx, iou in enumerate(cls_ious) \
                     if idx not in exclude_classes ]
        fractions = [ 100 * frac for idx, frac in enumerate(fractions) \
                     if idx not in exclude_classes ]
        cls_iou_strs = [ '%.2f'%(iou) for iou in cls_ious ] + \
                       [ '%.2f'%(speedup) ] + [ str(num_samples) ] + [ str(num_updates) ]
        csv_file.write(name + ',' + ','.join(cls_iou_strs) + '\n')
    #frac_strs = ["{0:.2f}".format(frac) for frac in fractions]
    #csv_file.write("Non-background pixel fraction," + ','.join(frac_strs) + '\n')
    csv_file.close()