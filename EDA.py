class EDA:
  
  def __init__(self, directory, sr=22050):
    self.directory = directory
    self.wav_name = os.listdir(directory)
    self.count = len(self.wav_name)
    self.file_duration = []
    for filename in tqdm(self.wav_name):
      signal, sr = librosa.load(os.path.join(self.directory,filename))
      duration = librosa.get_duration(signal, sr=sr) / 60
      self.file_duration.append(duration)
    self.df = pd.DataFrame(
        {"wav": self.wav_name, "duration(min)": self.file_duration},
        index=[index + 1 for index in range(self.count)]
        )
    
  def _summarize(self):
    return self.df[["duration(min)"]].describe()

  def _total_duration(self):
    hours = self.df["duration(min)"].sum() / 60
    return "total dataset duration: {} hours".format(str(hours))

  def _boxplot(self):
    sns.set_style("darkgrid")
    sns.set_palette("husl")
    fig, ax = plt.subplots()
    ax.set_title("Length")
    sns.boxplot(x=self.df["duration(min)"], ax=ax)
    plt.xticks(rotation=45)
    plt.show()
  
  def _violinplot(self):
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    fig, ax = plt.subplots()
    ax.set_title("Length")
    sns.violinplot(x=self.df["duration(min)"], ax=ax)
    plt.xticks(rotation=45)
    plt.show()