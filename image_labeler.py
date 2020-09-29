# WARNING: DO NOT RUN ON MACOS IT WILL CRASH THE WHOLE THING.

import argparse
import json
import os
import _tkinter
import tkinter as tk
from shutil import copyfile, move

import numpy as np
import pandas as pd
from PIL import Image, ImageTk

class Checkbar(tk.Frame):
    def __init__(self, parent=None, picks=None, side="left", anchor="w"):
        tk.Frame.__init__(self, parent)
        self.vars = []
        for pick in picks:
            var = tk.IntVar()
            chk = tk.Checkbutton(self, text=pick, variable=var)
            chk.pack(side=side, anchor=anchor, expand="yes")
            self.vars.append(var)
    
    def state(self):
        return map((lambda var: var.get()), self.vars)
    
    def clear(self):
        for var in self.vars:
            var.set(0)
        

class ImageGui:
    """
    GUI for iFind1 image sorting. This draws the GUI and handles all the events.
    Useful, for sorting views into sub views or for removing outliers from the data.
    """

    def __init__(self, master, labels, paths, configuration):
        """
            Initialise GUI of image labeler.
    
            Creates a GUI using Tkinter to help label images.
            NOTE: Tkinter has strange behavior if used in conjunction
            with Anaconda. The recommended route for using this program
            is through the default Python installation (Python 3.8.x).
            Otherwise, if Anaconda is used it will cause MacOS to crash.

            Paramters
            ---------
            master : Tkinter instance
                The root window that will contain the GUI
            labels : list(str)
                A list of labels that are associated with the images
            paths : list(str)
                A list of file paths to images
            configuration : dict
                Additional configurations (e.g., resizing)
                to smooth out user experience and allow for
                additional customization.
            
            Returns
            -------
            None.
        """

        # So we can quit the window from within the functions
        self.master = master
        self.configuration = dict()
        if isinstance(configuration, dict):
            self.configuration.update(configuration)
        else:
            raise TypeError("Configuration expected to be dict. Got {}".format(type(configuration)))
        # Extract the frame so we can draw stuff on it
        frame = tk.Frame(master)

        # Initialise grid
        frame.grid()

        # Start at the first file name
        self.index = 0
        self.paths = paths
        self.labels = labels
        #### added in version 2
        self.sorting_label = 'unsorted'
        ####

        # Number of labels and paths
        self.n_labels = len(labels)
        self.n_paths = len(paths)

        # Set empty image container
        self.image_raw = None
        self.image = None
        self.image_panel = tk.Label(frame)
        self.checkboxes = None
        self.buttons = []
        # set image container to first image
        self.set_image(paths[self.index])
        if self.configuration.get('multi_label_dataframe_location') is None:
            self.multi_label_data = []
        else:
            if os.path.exists(self.configuration.get('multi_label_dataframe_location')):
                try:
                    self.multi_label_data = pd.read_csv(self.configuration.get('multi_label_dataframe_location')).values.tolist()
                    self.index = len(self.multi_label_data)
                except pd.errors.EmptyDataError:
                    with open(self.configuration.get('multi_label_dataframe_location'), 'w') as fp: 
                        pass
                    self.multi_label_data = []
            else:
                with open(self.configuration.get('multi_label_dataframe_location'), 'w') as fp: 
                    pass
                self.multi_label_data = []
        # Add progress label
        progress_string = "%d/%d" % (self.index+1, self.n_paths)
        self.progress_label = tk.Label(frame, text=progress_string, width=10)
        # Place progress label in grid
        self.progress_label.grid(row=0, column=self.n_labels+2, sticky='we') # +2, since progress_label is placed after
                                                                            # and the additional 2 buttons "next im", "prev im"
        # Add sorting label
        sorting_string = os.path.split(df.sorted_in_folder[self.index])[-2]
        self.sorting_label = tk.Label(frame, text=("in folder: %s" % (sorting_string)), width=15)        
        # Place sorting label in grid
        self.sorting_label.grid(row=2, column=self.n_labels+1, sticky='we') # +2, since progress_label is placed after
                                                                            # and the additional 2 buttons "next im", "prev im"
            
        if self.configuration.get('is_multilabel'):
            print("Writing checkboxes")
            self.checkboxes = Checkbar(root, labels)
            self.checkboxes.grid(row=1, column=0, sticky='we')
            vote_button = tk.Button(frame, text="Submit", width=10, height=1, fg="blue", command=lambda: self.multi_label_vote(self.checkboxes))
            generate_button = tk.Button(frame, text="Generate", width=10, height=1, fg="blue", command=lambda: self.dataframe_to_csv(self.multi_label_data))
            vote_button.grid(row=3, column=0, sticky='we')
            generate_button.grid(row=3, column=1, sticky='we')
            self.buttons.append(tk.Button(frame, text="prev im", width=10, height=1, fg="green", command=self.move_prev_image))
            self.buttons.append(tk.Button(frame, text="next im", width=10, height=1, fg='green', command= self.move_next_image))
            tk.Label(frame, text="go to #pic:").grid(row=1, column=0)

            self.return_ = tk.IntVar() # return_-> self.index
            self.return_entry = tk.Entry(frame, width=6, textvariable=self.return_)
            self.return_entry.grid(row=1, column=1, sticky='we')
            master.bind('<Return>', self.num_pic_type)

            for ll, button in enumerate(self.buttons):
                button.grid(row=0, column=ll, sticky='we')
        else:
            # Make buttons
            for label in labels:
                self.buttons.append(
                        tk.Button(frame, text=label, width=10, height=2, fg='blue', command=lambda l=label: self.vote(l))
                )
                
            ### added in version 2
            self.buttons.append(tk.Button(frame, text="prev im", width=10, height=1, fg="green", command=self.move_prev_image))
            self.buttons.append(tk.Button(frame, text="next im", width=10, height=1, fg='green', command=self.move_next_image))
            ###
            # Place buttons in grid
            for ll, button in enumerate(self.buttons):
                button.grid(row=0, column=ll, sticky='we')
                #frame.grid_columnconfigure(ll, weight=1)
            #### added in version 2
            # Place typing input in grid, in case the mode is 'copy'
            if copy_or_move == 'copy':
                tk.Label(frame, text="go to #pic:").grid(row=1, column=0)

                self.return_ = tk.IntVar() # return_-> self.index
                self.return_entry = tk.Entry(frame, width=6, textvariable=self.return_)
                self.return_entry.grid(row=1, column=1, sticky='we')
                master.bind('<Return>', self.num_pic_type)
            ####
            
            # key bindings (so number pad can be used as shortcut)
            # make it not work for 'copy', so there is no conflict between typing a picture to go to and choosing a label with a number-key
            if copy_or_move == 'move':
                for key in range(self.n_labels):
                    master.bind(str(key+1), self.vote_key)

        # Place the image in grid
        self.image_panel.grid(row=2, column=0, columnspan=self.n_labels+1, sticky='we')

    def show_next_image(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index += 1
        progress_string = "%d/%d" % (self.index+1, self.n_paths)
        self.progress_label.configure(text=progress_string)
        
        #### added in version 2
        sorting_string = os.path.split(df.sorted_in_folder[self.index])[-2] #shows the last folder in the filepath before the file
        self.sorting_label.configure(text=("in folder: %s" % (sorting_string)))
        ####

        if self.index < self.n_paths:
            self.set_image(df.sorted_in_folder[self.index])
        else:
            self.master.quit()
    
    ### added in version 2        
    def move_prev_image(self):
        """
        Displays the prev image in the paths list AFTER BUTTON CLICK,
        doesn't update the progress display
        """
        self.index -= 1
        progress_string = "%d/%d" % (self.index+1, self.n_paths)
        self.progress_label.configure(text=progress_string)
        
        sorting_string = os.path.split(df.sorted_in_folder[self.index])[-2] #shows the last folder in the filepath before the file
        self.sorting_label.configure(text=("in folder: %s" % (sorting_string)))
        
        if self.index < self.n_paths:
            self.set_image(df.sorted_in_folder[self.index]) # change path to be out of df
        else:
            self.master.quit()
    
    ### added in version 2
    def move_next_image(self):
        """
        Displays the next image in the paths list AFTER BUTTON CLICK,
        doesn't update the progress display
        """
        self.index += 1
        progress_string = "%d/%d" % (self.index+1, self.n_paths)
        self.progress_label.configure(text=progress_string)
        sorting_string = os.path.split(df.sorted_in_folder[self.index])[-2] #shows the last folder in the filepath before the file
        self.sorting_label.configure(text=("in folder: %s" % (sorting_string)))
        
        if self.index < self.n_paths:
            self.set_image(df.sorted_in_folder[self.index])
        else:
            self.master.quit()

    def set_image(self, path):
        """
        Helper function which sets a new image in the image view
        :param path: path to that image
        """
        image = self._load_image(path)
        self.image_raw = image
        self.image = ImageTk.PhotoImage(image)
        self.image_panel.configure(image=self.image)

    def multi_label_vote(self, checkboxes):
        if df.sorted_in_folder[self.index] != df.im_path[self.index]:
            # if yes, use as input_path the current location of the image
            input_path = df.sorted_in_folder[self.index]
            root_ext, file_name = os.path.split(input_path)
            root, _ = os.path.split(root_ext)
        else:
            # if image hasn't been sorted use initial location of image
            input_path = df.im_path[self.index]
            root, file_name = os.path.split(input_path)
        #####
        checkbox_values = list(checkboxes.state())
        data = [os.path.normpath(input_path), checkbox_values]
        for value in checkbox_values:
            data.append(value)
        print(f"Appending Data {data} to index {self.index}")
        try:
            # If it's an index we've previously visited, modify directly
            self.multi_label_data[self.index] = data
        except IndexError:
            # It's a new instance, append the data.
            self.multi_label_data.append(data)
        
        checkboxes.clear()
        self.show_next_image()
      

    def vote(self, label):
        """
        Processes a vote for a label: Initiates the file copying and shows the next image
        :param label: The label that the user voted for
        """
        ##### added in version 2
        # check if image has already been sorted (sorted_in_folder != 0)
        if df.sorted_in_folder[self.index] != df.im_path[self.index]:
            # if yes, use as input_path the current location of the image
            input_path = df.sorted_in_folder[self.index]
            root_ext, file_name = os.path.split(input_path)
            root, _ = os.path.split(root_ext)
        else:
            # if image hasn't been sorted use initial location of image
            input_path = df.im_path[self.index]
            root, file_name = os.path.split(input_path)
        #####
        
        #input_path = self.paths[self.index]
        if copy_or_move == 'copy':
            self._copy_image(label, self.index)
        if copy_or_move == 'move':
            self._move_image(label, self.index)
            
        self.show_next_image()

    def vote_key(self, event):
        """
        Processes voting via the number key bindings.
        :param event: The event contains information about which key was pressed
        """
        pressed_key = int(event.char)
        label = self.labels[pressed_key-1]
        self.vote(label)
    
    #### added in version 2
    def num_pic_type(self, event):
        """Function that allows for typing to what picture the user wants to go.
        Works only in mode 'copy'."""
        # -1 in line below, because we want images bo be counted from 1 on, not from 0
        self.index = self.return_.get() - 1
        
        progress_string = "%d/%d" % (self.index+1, self.n_paths)
        self.progress_label.configure(text=progress_string)
        sorting_string = os.path.split(df.sorted_in_folder[self.index])[-2] #shows the last folder in the filepath before the file
        self.sorting_label.configure(text=("in folder: %s" % (sorting_string)))
        
        self.set_image(df.sorted_in_folder[self.index])

    def _load_image(self, path):
        """
        Loads and resizes an image from a given path using the Pillow library
        :param path: Path to image
        :return: Resized or original image 
        """
        resize = self.configuration.get('resize')
        image = Image.open(path)
        if resize:
            new_height = self.configuration.get('resize_height')
            new_width = self.configuration.get('resize_width')
            image = image.resize((new_height, new_width), Image.ANTIALIAS)
        else:
            max_height = 500
            img = image
            s = img.size
            ratio = max_height / s[1]
            image = img.resize((int(s[0]*ratio), int(s[1]*ratio)), Image.ANTIALIAS)
        return image

    @staticmethod
    def _copy_image(label, ind, df_path=None):
        """
        Copies a file to a new label folder using the shutil library. The file will be copied into a
        subdirectory called label in the input folder.
        :param input_path: Path of the original image
        :param label: The label
        """
        root, file_name = os.path.split(df.sorted_in_folder[ind])
        # two lines below check if the filepath contains as an ending a folder with the name of one of the labels
        # if so, this folder is being cut out of the path
        if os.path.split(root)[1] in labels:
            root = os.path.split(root)[0]
            os.remove(df.sorted_in_folder[ind])
            
        output_path = os.path.join(root, label, file_name)
        print("file_name =",file_name)
        print(" %s --> %s" % (file_name, label))
        copyfile(df.im_path[ind], output_path)
        
        # keep track that the image location has been changed by putting the new location-path in sorted_in_folder    
        df.loc[ind,'sorted_in_folder'] = output_path
        #####
        
        df.to_csv(df_path)

    @staticmethod
    def _move_image(label, ind):
        """
        Moves a file to a new label folder using the shutil library. The file will be moved into a
        subdirectory called label in the input folder. This is an alternative to _copy_image, which is not
        yet used, function would need to be replaced above.
        :param input_path: Path of the original image
        :param label: The label
        """
        root, file_name = os.path.split(df.sorted_in_folder[ind])
        # two lines below check if the filepath contains as an ending a folder with the name of one of the labels
        # if so, this folder is being cut out of the path
        if os.path.split(root)[1] in labels:
            root = os.path.split(root)[0]
        output_path = os.path.join(root, label, file_name)
        print("file_name =",file_name)
        print(" %s --> %s" % (file_name, label))
        move(df.sorted_in_folder[ind], output_path)
            
        # keep track that the image location has been changed by putting the new location-path in sorted_in_folder    
        df.loc[ind,'sorted_in_folder'] = output_path
        #####

    def dataframe_to_csv(self, dataframe_data=None):
        _column_names = ['file_name', 'label']
        _column_names.extend(self.labels)
        _temp_data_frame = pd.DataFrame(dataframe_data, columns=_column_names)
        _temp_data_frame.to_csv(self.configuration.get('multi_label_dataframe_location'), index=False)
        

def make_folder(directory=None):
    """
    Make folder if it doesn't already exist
    :param directory: The folder destination path
    """
    if directory is None:
        raise ValueError("Expected directory in form of file path. Received {}".format(directory))
    if not os.path.exists(directory):
        os.makedirs(directory, mode=0o775, exist_ok=True)

def load_configuration(file_path=None):
    json_data_structure = None
    if file_path is None:
        raise ValueError("Expected file path to JSON file. Received {}".format(file_path))
    if os.path.isdir(file_path):
        file_path = os.path.join(file_path, 'label_config.json')
    with open(file_path,'r') as _file:
        json_data_structure = json.load(_file)
    return json_data_structure

# The main bit of the script only gets exectured if it is directly called
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Apply labels to images provided.\n\
        This script is designed to quickly label images offline for use\
        in any image classification model. It is an extension on the work shown here\
        https://github.com/Nestak2/image-sorter2\n'
    )
    parser.add_argument(
        '-d',
        '--directory',
        dest='directory',
        help='The input directory where images to process are located.\
              For convenience, please use an absolute file path.\n\
              Default behavior is to check the current directory the script\
              is located in.',
        default=None,
        type=str
    )

    parser.add_argument(
        '-j',
        '--json',
        dest='json_file',
        help='The location of the JSON configuration file.\n\
            The default location is the current directory the script\
            is located in. JSON file name must be "label_config.json".',
        default=None,
        type=str
    )
    args = vars(parser.parse_args())
    if args['json_file'] is None:
        args['json_file'] = os.getcwd()
    configuration_dict = load_configuration(args['json_file'])
    if args['directory'] is None:
        args['directory'] = os.getcwd()
    else:
        args['directory'] = os.path.join(configuration_dict.get('frame_directory_root'), args['directory'])
    # Make folder for the new labels
    labels = configuration_dict.get('labels')
    for label in labels:
        make_folder(os.path.join(args['directory'], label))

    # Put all image file paths into a list
    ######## added in version 2
    file_names = [fn for fn in sorted(os.listdir(args['directory']))
                  if any(fn.lower().endswith(ext) for ext in configuration_dict.get('file_extensions'))]
    paths = [os.path.join(args['directory'], file_name) for file_name in file_names]
    
    copy_or_move = 'move'
    if copy_or_move == 'copy':
        try:
            df = pd.read_csv(configuration_dict.get("data_frame_location"), header=0)
            # Store configuration file values
        except FileNotFoundError:
            df = pd.DataFrame(columns=["im_path", 'sorted_in_folder'])
            df.im_path = paths
            df.sorted_in_folder = paths
    if copy_or_move == 'move':
        df = pd.DataFrame(columns=["im_path", 'sorted_in_folder'])
        df.im_path = paths
        df.sorted_in_folder = paths
    #######
    
# Start the GUI
root = tk.Tk()
app = ImageGui(root, labels, paths, configuration_dict)
root.mainloop()