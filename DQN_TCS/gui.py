import tkinter as tk
from tkinter import messagebox
import pygame
from pygame.locals import *
import torch
from PIL import Image, ImageTk
import tkinter as tk
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from agent import Agent

class GameApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 贪吃蛇游戏")

        # 加载背景图片
        self.background_image = Image.open("background.jpg")
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        self.background_label = tk.Label(root, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # 初始化pygame
        pygame.init()
        self.game_active = False
        self.game_paused = False

        # 设置游戏界面尺寸
        self.game = SnakeGameAI(w=640, h=480)
        self.agent = Agent()

        # 添加控件
        self.start_button = tk.Button(self.root, text="开始游戏", command=self.start_game)
        self.start_button.pack(side="bottom", pady=5)

        self.pause_button = tk.Button(self.root, text="暂停游戏", command=self.toggle_pause)
        self.pause_button.pack(side="bottom", pady=5)

        self.stop_button = tk.Button(self.root, text="结束游戏", command=self.stop_game)
        self.stop_button.pack(side="bottom", pady=5)
    def start_game(self):
        self.game_active = True
        self.game_paused = False
        self.update_game()

    def toggle_pause(self):
        if not self.game_active:
            return
        self.game_paused = not self.game_paused

    def stop_game(self):
        self.game_active = False
        final_score = self.game.score
        messagebox.showinfo("游戏结束", f"最终得分: {final_score}")
        pygame.quit()
        self.root.destroy()

    def update_game(self):
        if not self.game_active or self.game_paused:
            return

        # 更新游戏状态，从AI获取动作，更新游戏
        state_old = self.agent.get_state(self.game)
        final_move = self.agent.get_action(state_old)
        reward, done, score = self.game.play_step(final_move)
        state_new = self.agent.get_state(self.game)
        
        # 训练短期记忆
        self.agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # 记录数据
        self.agent.remember(state_old, final_move, reward, state_new, done)
        
        # 重置游戏
        if done:
            self.game.reset()
            self.agent.n_games += 1
            self.agent.train_long_memory()
            self.game_active = False
            messagebox.showinfo("游戏结束", f"最终得分: {score}")

        # 每50ms刷新游戏状态
        self.root.after(50, self.update_game)

def main():
    root = tk.Tk()
    root.geometry("640x480")  # 设置窗口大小
    app = GameApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()



