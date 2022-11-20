# -*- coding: utf8 -*-
'''
	DESCRIPTION
	Builds a content bundle so clients can ingest MOVs and test stickers with clips
'''

import os
import cv2
import glob
import shutil
import imutils
import ffmpeg
import pathlib
import macos_tags
from macos_tags import Tag, Color
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from lxml import etree
import importlib.util


class MakeHitAreaFromSequence(object):
	"""docstring for MakeHitAreaFromSequence.py"""
	def __init__(self, input_path=None, output_path=None):
		self.input_path = input_path
		self.output_path = output_path

		self.tc = self.module_from_file("tc", "text_colors.py")
		self.cu = self.module_from_file("cu", "catalog_utils.py")
		self.colors = self.tc.console
		self.cu.get_status()
		print("Running " + self.colors["BackgroundBlue"] + self.__class__.__name__  + self.colors["ENDC"])
		self.catalog_path = self.cu.get_catalog_path()
		self.clips_asset_dir = self.cu.get_clips_asset_dir()
		self.working_dir = self.cu.get_working_dir()
		self.media_dir = "Media"
		self.provided_poster_frame_dir = "PosterFrame"
		self.current_sequence_name = ""
		self.target_dir = ""
		self.length_in_millis = ""
		self.poster_frame_name_number = "_00000.png" # default, will try to find a better frame automatically
		self.poster_frame_index = 0 # this will be set automatically if we can match frames
		self.one_second = 5120
		

		# adjust this as needed. Lower numbers will capture more but also be less precise
		# whereas higher numbers will trace tighter, but somehow miss discontiguous areas.
		self.min_adjust = 8 

		# ---------- end user settings ---------- 

		
		self.current_length_of_sequence = ""
	
	
	def move_assets_to_content_bundle(self, incomingAssets):
		shutil.unpack_archive('__OverriddenContent__.zip', self.output_path)
		shutil.rmtree(os.path.join(self.input_path, '__MACOSX')) 
		assetNumber = 1

		def updateCat(seqUUID, motiUUID, fullLang, ISOLang):
			#img seq
			shutil.rmtree(os.path.join(self.output_path, "__OverriddenContent__/VideoAppsClipsAssets/Assets", seqUUID, "AssetData", fullLang)) 
			shutil.copytree(os.path.join(self.input_path, i, 'Media'), os.path.join(self.output_path, "__OverriddenContent__/VideoAppsClipsAssets/Assets", seqUUID, "AssetData", fullLang), dirs_exist_ok=True)
			#motion files
			shutil.rmtree(os.path.join(self.output_path, "__OverriddenContent__/VideoAppsClipsAssets/Assets", motiUUID, "AssetData", fullLang))
			shutil.copytree(os.path.join(self.input_path, i, 'PosterFrame'), os.path.join(self.output_path, "__OverriddenContent__/VideoAppsClipsAssets/Assets", motiUUID, "AssetData", fullLang, "Media"), dirs_exist_ok=True)
			shutil.copy2(os.path.join(self.input_path, i, i + ".moti"), os.path.join(self.output_path, "__OverriddenContent__/VideoAppsClipsAssets/Assets", motiUUID, "AssetData", fullLang, ISOLang + "_LoopStart.moti"))
			
		for i in incomingAssets:
			#remove errant poster frame from media folder
			os.remove(os.path.join(self.output_path, i, 'Media', i + '_Poster_Frame.png'))
			if assetNumber == 1:
				updateCat("8C1B0AA4-7665-49F3-82E0-0D2B805AD1DA", "A004F7F0-4298-47C8-87D6-5EBC7FBE0CBA", "English", "ENG")
			if assetNumber == 2:
				updateCat("88B3254C-EB00-4FC5-8D04-419DC6052FF9", "A48C9543-F059-424B-9CEE-FC4549144502", "Portuguese", "POR")
			if assetNumber == 3:
				updateCat("FA0F3A73-BD88-4C4A-A4B2-BD639D560717", "D6E6D694-6A1C-400C-9902-664F4790F738", "Spanish", "SPA")
			if assetNumber == 4:
				updateCat("F6E17C7C-7607-48B6-9A54-1200F57B3C4D", "3BCB0DDA-2A21-4DC3-8B37-1102BFCA2108", "French", "FRE")
			if assetNumber == 5:
				updateCat("1E0588AD-0ED2-4794-A724-643870A2E452", "516A5F32-9A58-4F6C-8E79-BD6F7038CCC2", "German", "GER")
			if assetNumber == 6:
				updateCat("EB03F152-2207-4EAE-BDF7-992ACA28317F", "7C1BF159-D820-4140-867D-DB66B75A4668", "Italian", "ITA")
			if assetNumber == 7:
				updateCat("B6447C99-1977-45E2-8C7F-132DC17D9139", "52885F43-502A-4DCA-A663-04E726AF83D1", "Arabic", "ARA")
			if assetNumber == 8:
				updateCat("3C519E03-0AD3-4653-B31F-2B1993B17FC6", "45720EE9-BAE7-4708-A4DA-2BEC890EF636", "Dutch", "DUT")
			if assetNumber == 9:
				updateCat("31E88359-AA16-44EB-A256-9AF9BA5846F0", "3E48A523-259D-4575-8A59-226A41B03948", "Japanese", "JAP")
			if assetNumber == 10:
				updateCat("6D728C56-DFAA-4641-B111-1DD11034B6E3", "9B68FEC1-F108-460A-9B6C-072B3E06ACFB", "Korean", "KOR")
			if assetNumber == 11:
				updateCat("4CAD6C3C-8F9B-48B1-A876-2B217B4E607E", "7A236B63-2AF1-451E-9820-8C0B9D12FE7A", "ChineseSimplified", "CHI_SIM")
			if assetNumber == 12:
				updateCat("86EB90FF-77F0-4D0A-A17B-D38DC06E01FF", "DBC51BB0-4DEB-4E46-9539-4A55CE618380", "ChineseTraditional", "CHI_TRAD")
			if assetNumber == 13:
				updateCat("7FF3676D-23A6-47D0-987E-A45C8BFA11A6", "B1610985-41D2-446A-9125-4FFB823C0B33", "Swedish", "SWE")
			if assetNumber == 14:
				updateCat("63B50799-A6A5-4609-9840-C41DE49A66DB", "3A5777E2-F27D-430E-901B-ECA25C09A49D", "Thai", "THA")
			if assetNumber == 15:
				updateCat("C9DF3B53-7279-4F3E-B911-595F2AF3EB7B", "23C8EC61-A0B0-42E8-B001-DC60CE8FC914", "Turkish", "TUR")
			if assetNumber == 16:
				updateCat("3719D08F-3B0A-4F07-9808-58FB9D87E04B", "34368F3D-97CD-4971-9C1F-45A9D3252757", "Vietnamese", "VIE")
			if assetNumber > 16:
				print("🛑 Only 16 stickers can be ingested at a time")
			assetNumber += 1

		print("🎉 __OverriddenContent__ bundle created! 🎉")
		#final cleanup, remove temp asset folders 
		subfolders = [ f.path for f in os.scandir(self.output_path) if f.is_dir() ]
		for directories in subfolders:
			if not "__OverriddenContent__" in directories:
				shutil.rmtree(directories)

	def video_to_frames(self):
		#first, clean up all folders
		subfolders = [f.path for f in os.scandir(self.output_path) if f.is_dir()]
		for directories in subfolders:
			shutil.rmtree(directories)
		assetList = []
		for file in os.listdir(self.input_path):
			if file.endswith(".mov"):
				print(os.path.join(self.input_path, file))
				os.makedirs(os.path.join(os.path.splitext(file)[0], 'Media'))
				os.makedirs(os.path.join(os.path.splitext(file)[0], 'PosterFrame'))

				input_file_name = file
				(ffmpeg
				.input(input_file_name)
				.filter('fps', fps=30, round = 'up')
				.filter('scale', 720, -1)
				.output(os.path.join(os.path.splitext(file)[0], 'Media', "%s_%%05d.png"%(input_file_name[:-4]), start_number=0, pix_fmt='rgba'))
				.run())
				
				assetList.append(os.path.splitext(file)[0])
				#count number of files so we can estimate the middle image for the poster frame, or use 100th frame if there's many
				numberOfFiles = next(os.path.join(os.walk(os.path.splitext(file)[0], 'Media')))[2]
				print(len(numberOfFiles))
				if len(numberOfFiles) <= 198:
					posterFrame = os.path.splitext(file)[0] + "_000" + str(int(len(numberOfFiles) / 2)) + ".png"
				else:
					posterFrame = os.path.splitext(file)[0] + "_00100.png"
				shutil.copy2(os.path.join(os.path.splitext(file)[0], 'Media', posterFrame), os.path.join(os.path.splitext(file)[0], 'PosterFrame', os.path.splitext(file)[0], "_Poster_Frame.png")) # complete target filename given

		return(assetList)


	def module_from_file(self, module_name, file_path):
		spec = importlib.util.spec_from_file_location(module_name, file_path)
		module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(module)
		return module
		

	def main(self):
		# print("already done")
		# return

	
		incomingAssets = self.video_to_frames()
		print(incomingAssets)
		#self.fix_dir_naming()
		#self.change_subdir_to_media_dir_name()
		#self.fix_sequence_zeros_and_compare_first_and_last_frames()
		#self.check_for_3_point_loop()

		# return
		for top_dir_name in os.listdir(self.input_path):
			self.target_dir = os.path.join(self.input_path, top_dir_name)
			self.current_sequence_name = top_dir_name

			poster_frame_exists = False
			provided_poster_frame_image = []
			poster_frame_name = ""

			# try to match the provided poster frame with one of the sequence items
			for file_name in glob.glob(os.path.join(self.target_dir, self.provided_poster_frame_dir, "*.png")):
				print("I found a provided poster frame: " + file_name)
				provided_poster_frame_image = cv2.imread(file_name, cv2.IMREAD_COLOR)

			file_count = 0
			for file_name in glob.glob(os.path.join(self.target_dir, self.media_dir, "*.png")):
				if "Poster_Frame" in file_name:
					poster_frame_exists = True
					print("Poster frame already exists")
				else:
					file_count += 1

			self.current_length_of_sequence = str(file_count)
			
			# print("Length of sequence: " + str(self.current_length_of_sequence))
			self.length_in_millis = str(  (self.one_second*int(self.current_length_of_sequence))-(self.one_second )  )

			# 
			for root, dirs, files in os.walk(os.path.join(self.target_dir, self.media_dir)):
					
				for f in files:
					if len(provided_poster_frame_image)>0:

						# compare the provide frame with each sequence frame until you find it
						# you need to do it this way so you know which frame number to use in the motion project
						tmp_frame = cv2.imread(os.path.join(self.target_dir, self.media_dir, f), cv2.IMREAD_COLOR)
						
						
						# print(np.array_equiv(tmp_frame, provided_poster_frame_image))

						if np.array_equiv(tmp_frame, provided_poster_frame_image):
							print("I found a frame that matches the provided poster: " + f)

							# todo refactor this 
							poster_frame_name = f[0:f.rindex("_00")]
							self.current_sequence_name = poster_frame_name
							print("\tcurrent_sequence_name: '" + poster_frame_name + "'")
							poster_frame_name += "_Poster_Frame.png"
							path_to_frame = os.path.join(self.target_dir, self.media_dir, poster_frame_name)
							path_to_frame = path_to_frame.replace(" ", "\ ")
							arg = "cp " + os.path.join(self.target_dir, self.media_dir, f) + " " + path_to_frame
							os.system(arg)

							# finally, set this so we can set the marker in xml
							slice_start = f.rindex("_")
							slice_end = f.rindex(".png")

							self.poster_frame_index = int(f[slice_start+1:slice_end])

							break

					elif self.poster_frame_name_number in f: # this must be the sequence
						poster_frame_name = f[0:f.rindex("_00")]
						self.current_sequence_name = poster_frame_name
						print("\tcurrent_sequence_name: '" + poster_frame_name + "'")
						poster_frame_name += "_Poster_Frame.png"
						path_to_frame = os.path.join(self.target_dir, self.media_dir, poster_frame_name)
						path_to_frame = path_to_frame.replace(" ", "\ ")
						arg = "cp " + os.path.join(self.target_dir, self.media_dir, f) + " " + path_to_frame
						os.system(arg)

				if poster_frame_name == "":
					print("⚠️ Pick a suitable frame and place it in the poster frame folder.")
				else:
					print("\tposter_frame_name: '" + poster_frame_name + "'")
					self.get_hit_area_points(os.path.join(self.target_dir, "Media", poster_frame_name))

		self.move_assets_to_content_bundle(incomingAssets)

	def fix_sequence_zeros_and_compare_first_and_last_frames(self):
		firstFrame = ""
		lastFrame = ""
		print(self.colors["BackgroundBlue"] + "Renaming the file to use 00000 sequence format..." + self.colors["ENDC"])
		for top_dir_name in os.listdir(self.input_path):
			if os.path.isdir(os.path.join(self.input_path, top_dir_name)):
				self.target_dir = os.path.join(self.input_path, top_dir_name)
				# get number of PNGs in the current media folder
				pngCounter = len(glob.glob1(os.path.join(self.target_dir, self.media_dir), "*.png"))
				i=0
				while i < pngCounter:
					for file_name in sorted(glob.glob(os.path.join(self.target_dir, self.media_dir, "*.png"))):
						# removes numerals from filename
						res = ''.join(i for i in file_name if not i.isdigit())
						# inserts numbers to filename (zfill always ensures correct amount of leading zeroes (5 digits total))
						new_file_name = res[:len(res)-4] + str(i).zfill(5) + res[len(res)-4:]
						if i == 0:
							firstFrame = new_file_name
							#print(self.target_dir + " PNG Counter = " + str(pngCounter))
						i += 1
						if "Poster_Frame" not in new_file_name:
							lastFrame = new_file_name
							arg = "mv " + file_name + " " + new_file_name
							os.system(arg)


				firstFrame = cv2.imread(firstFrame)
				lastFrame = cv2.imread(lastFrame)

				# Convert images to grayscale
				first_gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
				last_gray = cv2.cvtColor(lastFrame, cv2.COLOR_BGR2GRAY)
				
				# Compute SSIM (structural similarity index) between two images
				score, diff = structural_similarity(first_gray, last_gray, full=True)
				print(self.target_dir + " First & Last Frame Similarity Score: {:.3f}%".format(score * 100))

				# determing potential 3-point-loop and adding or removing tag to folder
				tag = Tag(name="3_Point_Loop", color=Color.ORANGE)
				if (score * 100) < 60:
					#print(self.target_dir + " is a potential 3 Point Loop")
					macos_tags.add(tag, file=self.target_dir)
				else:
					macos_tags.remove(tag, file=self.target_dir)



	def fix_dir_naming (self):
		print(self.colors["BackgroundBlue"] + "Fixing dir names..." + self.colors["ENDC"])
		for top_dir_name in os.listdir(self.input_path):
			
			if os.path.isdir(os.path.join(self.input_path, top_dir_name)):

				if " " in top_dir_name or "_" in top_dir_name:
					#removes spaces and underscores and capitalizes each word (pascal case)
					new_top_dir_name = top_dir_name.replace("_", " ").title().replace(" ", "")
					top_dir_name = top_dir_name.replace(" ", "\ ")
					arg = "mv " + os.path.join(self.input_path, top_dir_name) + " " + os.path.join(self.input_path, new_top_dir_name)
					os.system(arg)
					print("\tDir name updated --> " + new_top_dir_name)
				# add other situations here...
				if "%" in top_dir_name:
					new_top_dir_name = top_dir_name.replace("%", "PCNT")
					arg = "mv " + os.path.join(self.input_path, top_dir_name) + " " + os.path.join(self.input_path, new_top_dir_name)
					os.system(arg)
					print("\tDir name updated --> " + new_top_dir_name)

	def change_subdir_to_media_dir_name(self):
		print(self.colors["BackgroundBlue"] + "Renaming sequence dirs to 'Media'" + self.colors["ENDC"])
		for top_dir_name in os.listdir(self.input_path):
			if os.path.isdir(os.path.join(self.input_path, top_dir_name)):
				for sub_dir_name in os.listdir(self.input_path + top_dir_name + "/"):
					if "PNG Sequence" in sub_dir_name or "PNG" in sub_dir_name:
						arg = "mv " + os.path.join(self.input_path, top_dir_name, sub_dir_name.replace(" ", "\ ")) + " " + os.path.join(self.input_path, top_dir_name, "Media")
						os.system(arg)
			


	def get_hit_area_points(self, path_to_frame):

		path_to_frame = path_to_frame.replace("\ ", " ")
		if not os.path.isfile(path_to_frame):
			print("\t❌ PNG doesn't exist. " + path_to_frame)
			raise RuntimeError from None

		# traced image is a way to get outlines that are a little thicker
		# adjust the 0x1 to be 0x3 or higher for larger boundaries
		traced_image = path_to_frame.replace(".png", "-hit.png")
		arg = "convert " + path_to_frame + " -alpha extract -blur 0x1 -transparent black -channel RGB -negate -alpha extract -threshold 0 -negate -transparent white " + traced_image
		os.system( arg )

		# image = cv2.imread(path_to_frame, cv2.IMREAD_GRAYSCALE) # revert to this method if you don't want the traced_image technique
		image = cv2.imread(traced_image, cv2.IMREAD_GRAYSCALE)
		color_image = cv2.imread(path_to_frame, cv2.IMREAD_COLOR)
		color_image = cv2.resize(color_image, (1080, 1080))
		image = cv2.resize(image, (1080, 1080))

		# delete traced image from file system if it exists
		if os.path.isfile(traced_image):
			arg = "rm " + traced_image
			os.system( arg )

		# gray = cv2.GaussianBlur(image, (0, 0), 5) # 12 is best middle of the road. 
		gray = cv2.GaussianBlur(image, (0, 0), 3)

		# 1st attempt 
		thresh = self.get_threshold_image(gray, "") #try getting contours using regular binary method

		# find contours in thresholded image, then grab the largest one
		hull = self.obtain_contours(gray, thresh) 

		# 2nd attempt if we didn't get any contours.....some images need this thresh inversion
		if len(hull) < 10: # need to redo the threshold using invert method
			print("\t--> will try to get a better contour...")
			thresh = self.get_threshold_image(gray, "inverse") #try getting contours using regular binary method

			# find contours in thresholded image, then grab the largest one
			hull = self.obtain_contours(gray, thresh) 

		reduced_contours = []
		precision_amount = 2
		if len(hull) <= 30:
			precision_amount = 1
		elif len(hull) <= 50:
			precision_amount = 2
		elif len(hull) <= 80:
			precision_amount = 3
		elif len(hull) <= 120:
			precision_amount = 4
		elif len(hull) <= 150:
			precision_amount = 5
		elif len(hull) <= 180:
			precision_amount = 6
		elif len(hull) > 180:
			precision_amount = 10

		counter = 0
		for contour_item in hull:
			if counter % precision_amount == 0:
				reduced_contours.append( contour_item )
				# print(contour_item)
			counter += 1
		reduced_contours = np.asarray(reduced_contours)
		
		print("\tReduced contour now has " + str(len( reduced_contours )) + " points")
		cv2.drawContours(color_image, contours=[reduced_contours], contourIdx=0, color=(0, 255, 255), thickness=2)
		
		#cv2.imshow("Hit area trace with " + str(len( reduced_contours )) + " points", color_image)
		#cv2.waitKey(0)

		self.construct_xml(reduced_contours.tolist())

	def obtain_contours(self, gray, thresh):
		contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = imutils.grab_contours(contours)
		print("\tnumber of shapes is " + str( len(contours) ))

		if len(contours) == 0:
			print(self.colors["ERROR_EMOJI"] + " Having a problem getting points for " + self.colors["BackgroundRed"] + path_to_frame + self.colors["ENDC"])
			print("no counters")
			return 

		# merge contours if necessary
		if len(contours) > 1:
			list_of_pts = [] 
			for ctr in contours:
				list_of_pts += [pt[0] for pt in ctr]
			reshaped_contours = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)
			hull = cv2.convexHull( reshaped_contours )
		else:
			hull = cv2.convexHull(contours[0])
		
		img_hull = cv2.drawContours(gray, contours=[hull], contourIdx=0, color=(255, 0, 0), thickness=2)

		# cv2.imshow("Hull", img_hull)
		# cv2.waitKey(0)
		print('\toriginal hull contours length: ' + str(len(hull)))

		return hull

	def get_threshold_image(self, gray, mode):
		if mode == "inverse":
			print("\tUsing cv2.THRESH_BINARY_INV")
			thresh = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY_INV)[1]
		else:
			print("\tUsing cv2.THRESH_BINARY")
			thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

		return thresh

	def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
		"""Return a sharpened version of the image, using an unsharp mask."""
		blurred = cv2.GaussianBlur(image, kernel_size, sigma)
		sharpened = float(amount + 1) * image - float(amount) * blurred
		sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
		sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
		sharpened = sharpened.round().astype(np.uint8)
		if threshold > 0:
			low_contrast_mask = np.absolute(image - blurred) < threshold
			np.copyto(sharpened, image, where=low_contrast_mask)
		return sharpened

	def construct_xml(self, list_of_coords):

		# set up nodes to create
		curve_X = Element('curve_X')
		curve_Y = Element('curve_Y')
		
		# set comments
		comment = Comment('Hit area has been autogenerated.')
		curve_X.append(comment)
		curve_Y.append(comment)
		
		# print(len(list_of_coords))
		vertex_id = 0 # needs to be unique at vertex level

		for coord in list_of_coords:
			# print( coord[0][0], coord[0][1] )
			x_value = coord[0][0]
			y_value = coord[0][1]

			# transform into the motion coord system
			x_value = x_value - 540
			y_value = y_value - 540

			# for some reason the shape is coming in backward, so this fixes that:
			# x_value = x_value * -1
			y_value = y_value * -1

			# vertex_id = str(list_of_coords.index(coord)+1)
			vertex_folder_flag = "8590004240" #hard-coding for now, Motion can override w/o problem
			parameter_flag = "12884967440" #hard-coding for now, Motion can override w/o problem
			
			# X COORD
			vertex_x = SubElement(curve_X, 'vertex',
										{'index':str(list_of_coords.index(coord)),
										'id':str(vertex_id),
										'flags':str(136),
										}
										)

			vertex_x_folder = SubElement(vertex_x, 'vertex_folder',
										{'name':'Vertex',
										'id':str(vertex_id),
										'flags':vertex_folder_flag,
										}
										)

			parameter_x = SubElement(vertex_x_folder, 'parameter',
										{'name':'Value',
										'id': str(2),
										'flags':parameter_flag,
										'default':'0',
										'value': str(x_value),
										}
										)
			vertex_id +=1

			# Y COORD
			vertex_y = SubElement(curve_Y, 'vertex',
										{'index':str(list_of_coords.index(coord)),
										'id':str(vertex_id),
										'flags':str(136),
										}
										)

			vertex_y_folder = SubElement(vertex_y, 'vertex_folder',
										{'name':'Vertex',
										'id':str(vertex_id),
										'flags':vertex_folder_flag,
										}
										)

			parameter_y = SubElement(vertex_y_folder, 'parameter',
										{'name':'Value',
										'id': str(2),
										'flags':parameter_flag,
										'default':'0',
										'value': str(y_value),
										}
										)

		self.append_node_to_XML(curve_X, curve_Y)

	def append_node_to_XML(self, curve_X, curve_Y):
		# print(tostring(node))
		parser = etree.XMLParser(remove_blank_text=True)
		tree = etree.parse('./template.moti', parser)
		root = tree.getroot()

		# ====================================
		# update the sequence frame
		# ====================================

 		# <clip name="sequence" id="10115">
  		#	   <relativeURL>Media/SI020_ValenD_flamingHeart_A_v08_%5B%23%23%23%23%23%5D.png:0:59</relativeURL>
		node_to_remove = tree.xpath("//clip[@name='sequence']/relativeURL") 

		# for some reason this is a list of 1
		node_to_remove[0].getparent().remove(node_to_remove[0])

		# add back a fresh node
		node_to_add = tree.xpath("//clip[@name='sequence']")
		etree.SubElement(node_to_add[0], "relativeURL").text = "Media/" + self.current_sequence_name + "_%5B%23%23%23%23%23%5D.png:0:" + str(int(self.current_length_of_sequence)-1)

		# ====================================
		# update the poster sequence
		# ====================================
		# <clip name="Poster Frame" id="10118">
 		#	   <relativeURL>Media/SI020_ValenD_flamingHeart_A_v08_Poster%20Frame.png</relativeURL>
		node_to_remove = tree.xpath("//footage[@name='Media Layer']/clip[@name='Poster Frame']/relativeURL") 

		# for some reason this is a list of 1
		node_to_remove[0].getparent().remove(node_to_remove[0])
		
		# add back a fresh node
		node_to_add = tree.xpath("//footage[@name='Media Layer']/clip[@name='Poster Frame']")
		etree.SubElement(node_to_add[0], "relativeURL").text = "Media/" + self.current_sequence_name + "_Poster_Frame.png"

		# timing/duration of the poster frame
		# <scenenode name="Poster Frame" id="10119" factoryID="7" version="5">
		# 	<validTracks>1</validTracks>
		# 	<aspectRatio>1</aspectRatio>
		# 	<flags>0</flags>
		# 	<timing in="0 1 1 0" out="302080 153600 1 0" offset="0 1 1 0"/>
		
		node_to_append = tree.xpath("//scenenode[@name='Poster Frame']") 
		node_to_remove = tree.xpath("//scenenode[@name='Poster Frame']/timing") 
		# <timing in="0 1 1 0" out="419840 153600 1 0" offset="0 1 1 0"/>
		node_to_remove[0].getparent().remove(node_to_remove[0])
		timing_node = Element('timing', {"in":"0 1 1 0", "out": self.length_in_millis + " 153600 1 0", "offset":'0 1 1 0'})
		node_to_append[0].insert( 4, etree.XML(tostring(timing_node)) )

		# <scene>
		# 	<sceneSettings>
		# 		<width>1080</width>
		# 		<height>1080</height>
		# 		<duration>72</duration>
		node_to_append = tree.xpath("//scene/sceneSettings") 
		node_to_remove = tree.xpath("//scene/sceneSettings/duration") 
		node_to_remove[0].getparent().remove(node_to_remove[0])
		duration_node = Element('duration')
		duration_node.text = str( int(self.current_length_of_sequence) )
		node_to_append[0].insert( 2, etree.XML(tostring(duration_node)) )

		# media layer poster frame
		node_to_append = tree.xpath("//footage[@name='Media Layer']/clip[@name='Poster Frame']") 
		node_to_remove = tree.xpath("//footage[@name='Media Layer']/clip[@name='Poster Frame']/timing") 
		node_to_remove[0].getparent().remove(node_to_remove[0])

		timing_node = Element('timing', {"in":"0 1 1 0", "out": self.length_in_millis + " 153600 1 0", "offset":'0 1 1 0'})
		node_to_append[0].insert( 4, etree.XML(tostring(timing_node)) )

		# update the footage timing
		# <footage name="Media Layer"
		# 	<timing in="0 1 1 0" out="424960 153600 1 0" offset="0 1 1 0"/>
		node_to_append = tree.xpath("//footage[@name='Media Layer']") 
		node_to_remove = tree.xpath("//footage[@name='Media Layer']/timing") 
		node_to_remove[0].getparent().remove(node_to_remove[0])
		timing_node = Element('timing', {"in":"0 1 1 0", "out": self.length_in_millis + " 153600 1 0", "offset":'0 1 1 0'})
		node_to_append[0].insert( 4, etree.XML(tostring(timing_node)) )

		# ====================================
		# add the hit area
		# ====================================
		node_to_append = tree.xpath("//layer[@name='Sequence']/scenenode[@name='Hit Area']/parameter[@name='Object']/parameter[@name='Shape']/parameter[@name='Animation']") 
		node_to_append[0].insert( 0, etree.XML(tostring(curve_Y)) )
		node_to_append[0].insert( 0, etree.XML(tostring(curve_X)) )

		# time duration of the hit area
		# <timing in="0 153600 1 0" out="424960 153600 1 0" offset="20480 153600 1 0"/>
		# 430080   424960
		node_to_append = tree.xpath("//layer[@name='Sequence']/scenenode[@name='Hit Area']") 
		node_to_remove = tree.xpath("//layer[@name='Sequence']/scenenode[@name='Hit Area']/timing") 
		node_to_remove[0].getparent().remove(node_to_remove[0])
		timing_node = Element('timing', {"in":"0 1 1 0", "out": self.length_in_millis + " 153600 1 0", "offset":'0 1 1 0'})
		node_to_append[0].insert( 4, etree.XML(tostring(timing_node)) )


		# update the containing Sequence layer duration
		node_to_append = tree.xpath("//layer[@name='Sequence']")
		node_to_remove = tree.xpath("//layer[@name='Sequence']/timing") 
		node_to_remove[0].getparent().remove(node_to_remove[0])
		timing_node = Element('timing', {"in":"0 1 1 0", "out": self.length_in_millis + " 153600 1 0", "offset":'0 1 1 0'})
		node_to_append[0].insert( 4, etree.XML(tostring(timing_node)) )

		# ====================================
		# remove the out attribute from seq
		# ====================================
		# <scenenode name="Image Sequence"
		# <timing in="0 1 1 0" out="37760000 19200000 3 0" offset="0 1 1 0"/>
		node_to_remove = tree.xpath("//scenenode[@name='Image Sequence']/timing") 
		node_to_remove[0].getparent().remove(node_to_remove[0])

		node_to_append = tree.xpath("//scenenode[@name='Image Sequence']")
		timing_node = Element('timing', {"in":"0 1 1 0", "out": self.length_in_millis + " 153600 3 0", "offset":'0 1 1 0'})
		node_to_append[0].insert( 5, etree.XML(tostring(timing_node)) )

		# ====================================
		# update the timing node (found under the media tab)
		# ====================================
		# <scenenode name="Image Sequence"
		# 		<parameter name="Properties" 
		# 				<parameter name="Retime Value" 
		# 						<curve 
		# 							the second keypoint!!!
		# 							<keypoint interpolation="1" flags="0">
		# 								<time>5120*LENGTH 153600 1 0</time>
		# 								<value>2</value>
		# 							</keypoint>
		# value node
		node_to_remove = tree.xpath("//scenenode[@name='Image Sequence']/parameter[@name='Properties']/parameter[@name='Retime Value']/curve/keypoint[2]/value") 
		tmp_parent = node_to_remove[0].getparent()
		node_to_remove[0].getparent().remove(node_to_remove[0])
		value_node = Element('value')
		# print(self.current_length_of_sequence )
		value_node.text = str( int(self.current_length_of_sequence) + 1 )
		tmp_parent.insert( 0, etree.XML(tostring(value_node)) )

		# time node
		node_to_remove = tree.xpath("//scenenode[@name='Image Sequence']/parameter[@name='Properties']/parameter[@name='Retime Value']/curve/keypoint[2]/time") 
		tmp_parent = node_to_remove[0].getparent()
		node_to_remove[0].getparent().remove(node_to_remove[0])
		time_node = Element('time')
		time_node.text = str(int(self.length_in_millis) + self.one_second) + " 153600 1 0"
		tmp_parent.insert( 0, etree.XML(tostring(time_node)) )

		# do the same for the Retime Value Cache
		# value node
		node_to_remove = tree.xpath("//scenenode[@name='Image Sequence']/parameter[@name='Properties']/parameter[@name='Retime Value Cache']/curve/keypoint[2]/value") 
		tmp_parent = node_to_remove[0].getparent()
		node_to_remove[0].getparent().remove(node_to_remove[0])
		value_node = Element('value')
		value_node.text = str(self.current_length_of_sequence)
		tmp_parent.insert( 0, etree.XML(tostring(value_node)) )

		# time node
		node_to_remove = tree.xpath("//scenenode[@name='Image Sequence']/parameter[@name='Properties']/parameter[@name='Retime Value Cache']/curve/keypoint[2]/time") 
		tmp_parent = node_to_remove[0].getparent()
		node_to_remove[0].getparent().remove(node_to_remove[0])
		time_node = Element('time')
		time_node.text = self.length_in_millis + " 153600 1 0"
		tmp_parent.insert( 0, etree.XML(tostring(time_node)) )

		# add this to first curve <postExtrapolation>1</postExtrapolation>
		# <parameter name="Retime Value" id="304" flags="8590066066">
		# 			<curve type="1" default="1" value="1" round="0" retimingExtrapolation="1">

		# it should look like this:
		# <parameter name="Retime Value" id="304" flags="8590066066">
		# 	<curve type="1" default="1" value="1" round="0" retimingExtrapolation="1">
		# 		<numberOfKeypoints>2</numberOfKeypoints>
		# 		<postExtrapolation>1</postExtrapolation>
		# 		<keypoint interpolation="1" flags="0">
		# 			<time>0 1 1 0</time>
		# 			<value>1</value>
		# 		</keypoint>
		# 		<keypoint interpolation="1" flags="0">
		# 			<time>430080 153600 1 0</time>
		# 			<value>85</value>
		# 		</keypoint>
		# 	</curve>
		# </parameter>
		node_to_append = tree.xpath("//scenenode[@name='Image Sequence']/parameter[@name='Properties']/parameter[@name='Retime Value']/curve") 
		post_extrapolation_node = Element('postExtrapolation')
		post_extrapolation_node.text = "1"
		node_to_append[0].insert( 1, etree.XML(tostring(post_extrapolation_node)) )


		# update the duration cache
		# <parameter name="Duration Cache" id="320" flags="8589934610" default="0" value="85"/>
		node_to_remove = tree.xpath("//scenenode[@name='Image Sequence']/parameter[@name='Properties']/parameter[@name='Duration Cache']") 
		tmp_parent = node_to_remove[0].getparent()
		node_to_remove[0].getparent().remove(node_to_remove[0])
		duration_cache_node = Element('parameter', {"name":"Duration Cache", "id": "320", "flags":"8589934610", "default":"0", "value":str(self.current_length_of_sequence)})
		tmp_parent.insert( 7, etree.XML(tostring(duration_cache_node)) )


		# <layer name="Background"
		# 			<timing in="0 1 1 0" out="222560 153600 1 0" offset="0 1 1 0"/>
		node_to_remove = tree.xpath("//layer[@name='Background']/timing") 
		tmp_parent = node_to_remove[0].getparent()
		node_to_remove[0].getparent().remove(node_to_remove[0])
		timing_node = Element('timing', {"in":"0 1 1 0", "out": self.length_in_millis + " 153600 1 0", "offset":'0 1 1 0'})
		tmp_parent.insert( 4, etree.XML(tostring(timing_node)) )

		# layer --> Title Background
		# <layer name="Background" id="10003">
  		#     	<scenenode name="Title Background" id="10007" factoryID="7" version="5">
		node_to_remove = tree.xpath("//layer[@name='Background']/scenenode[@name='Title Background']/timing") 
		tmp_parent = node_to_remove[0].getparent()
		node_to_remove[0].getparent().remove(node_to_remove[0])
		timing_node = Element('timing', {"in":"0 1 1 0", "out": self.length_in_millis + " 153600 1 0", "offset":'0 1 1 0'})
		tmp_parent.insert( 4, etree.XML(tostring(timing_node)) )

		# <clip name="Title Background"	
		# 			<timing in="0 1 1 0" out="222560 153600 1 0" offset="0 1 1 0"/>
		# node_to_remove = tree.xpath("//clip[@name='Title Background']/timing") 
		# tmp_parent = node_to_remove[0].getparent()
		# node_to_remove[0].getparent().remove(node_to_remove[0])
		# timing_node = Element('timing', {"in":"0 1 1 0", "out": self.length_in_millis + " 153600 1 0", "offset":'0 1 1 0'})
		# tmp_parent.insert( 4, etree.XML(tostring(timing_node)) )


		# ====================================
		# update the creation duration
		# ====================================
		# <clip name="sequence" id="10115">
		# 	<relativeURL>Media/ReStickIt_Earthday_Earth_02_720px_%5B%23%23%23%23%23%5D.png:0:119</relativeURL>
		# 	<creationDuration>120</creationDuration>
		node_to_remove = tree.xpath("//clip[@name='sequence']/creationDuration") 
		tmp_parent = node_to_remove[0].getparent()
		tmp_parent.remove(node_to_remove[0])
		creationDuration_node = Element('creationDuration')
		creationDuration_node.text = str( int(self.current_length_of_sequence) )
		tmp_parent.insert( 0, etree.XML(tostring(creationDuration_node)) )

		# <clip name="sequence" id="10115">
		#  		<timing in="0 1 1 0" out="37760000 19200000 3 0" offset="0 1 1 0"/>
		node_to_remove = tree.xpath("//clip[@name='sequence']/timing") 
		tmp_parent = node_to_remove[0].getparent()
		tmp_parent.remove(node_to_remove[0])
		timing_node = Element('timing', {"in":"0 1 1 0", "out": self.length_in_millis + " 153600 1 0", "offset":'0 1 1 0'})
		tmp_parent.insert( 8, etree.XML(tostring(timing_node)) )

		# ====================================
		# set the project duration
		# ====================================
		# one frame = 5120
		# <scene>
		# 	<timeRange offset="0 1 1 0" duration="1307200 153600 1 0"/>
		#   <playRange offset="0 1 1 0" duration="1307200 153600 1 0"/>

		# timeRange
		node_to_remove = tree.xpath("//scene/timeRange") 
		tmp_parent = node_to_remove[0].getparent()
		tmp_parent.remove(node_to_remove[0])

		node_to_add = Element('timeRange', {"foo":'0', "offset":'0 1 1 0', "duration": str( int(self.length_in_millis) + self.one_second ) + ' 153600 1 0'})
		tmp_parent.insert( 0, etree.XML(tostring(node_to_add)) )

		# playRange
		node_to_remove = tree.xpath("//scene/playRange") 
		tmp_parent = node_to_remove[0].getparent()
		tmp_parent.remove(node_to_remove[0])

		node_to_add = Element('playRange', {"offset":'0 1 1 0', "duration": str( int(self.length_in_millis) + self.one_second ) + ' 153600 1 0'})
		tmp_parent.insert( 0, etree.XML(tostring(node_to_add)) )

		# ====================================
		# move the ending loop-end marker to the end of the timeline
		# ====================================
		# <timemarkerset>
		# 	<timemarker>
		# 		<inpoint>148480 153600 1 0</inpoint>
		# 		<color>1</color>
		# 		<type>8</type>
		# 	</timemarker>
		# 	<timemarker>
		# 		<inpoint>307200 153600 1 0</inpoint>
		# 		<color>1</color>
		# 		<type>7</type>
		# 	</timemarker>
		# </timemarkerset>

		# 1st node - this is the poster frame, don't move it for now 
		node_to_remove = tree.xpath("//timemarkerset/timemarker/inpoint") 
		tmp_parent = node_to_remove[0].getparent()
		tmp_parent.remove(node_to_remove[0])

		node_to_add = Element('inpoint')
		node_to_add.text = str(5120*int(self.poster_frame_index)) + " 153600 1 0"
		tmp_parent.insert( 0, etree.XML(tostring(node_to_add)) )

		# 2nd node - the loop end
		node_to_remove = tree.xpath("//timemarkerset/timemarker/inpoint") 
		tmp_parent = node_to_remove[1].getparent()
		tmp_parent.remove(node_to_remove[1])

		node_to_add = Element('inpoint')
		node_to_add.text = str( int(self.length_in_millis) + self.one_second ) + " 153600 1 0"
		tmp_parent.insert( 0, etree.XML(tostring(node_to_add)) )

		# ====================================
		# save the motion file
		# ====================================
		print("✅ Writing a new motion file: " + self.current_sequence_name + '.moti')
		tree.write(self.target_dir + self.current_sequence_name + '.moti', pretty_print=True)

		# ====================================
		# clean up
		# ====================================
		del tree
		del parser 

# run the script
if __name__ == '__main__':
	runner = MakeHitAreaFromSequence()
	runner.main()
# print(BLINK + "Have a nice day." + self.colors["ENDC"])
