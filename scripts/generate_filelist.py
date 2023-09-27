import os


if __name__ == "__main__":
  folderroot = "projects/torchdnn/data/dcmt/ChasingDrones"
  
  filenames = []
  for filename in os.listdir(folderroot):
    filepath = os.path.join(folderroot, filename)
    if os.path.isfile(filepath):
      filenames.append(filename)
  
  filenames = sorted(filenames)
  
  savepath = os.path.join(folderroot, "filelist.txt")
  with open(savepath, 'w') as f:
    for filename in filenames:
      f.write(filename + '\n')
    
    
  
  