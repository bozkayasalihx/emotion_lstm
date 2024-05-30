import torch 
import torch.nn as nn

class EmotionLSTM(nn.Module):
    def __init__(self, num_of_emotions):
        super().__init__()
        self.conv2Dblock = nn.Sequential(
            ### 1. conv2d block
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1, padding=1,),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            
            ### 2. conv2d block
            nn.Conv2d( in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3. conv block
            nn.Conv2d( in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 4. conv block
            nn.Conv2d( in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        # LSTM block
        self.lstm_maxpool = nn.MaxPool2d(kernel_size=[2, 4], stride=[2, 4])

        hidden_size = 128
        self.lstm = nn.LSTM( input_size=64, hidden_size=hidden_size, bidirectional=True, batch_first=True,)

        self.dropout_lstm = nn.Dropout(0.1)
        self.attention_linear = nn.Linear( 2 * hidden_size, 1,)
        # Linear softmax layer
        self.out_linear = nn.Linear(4 * hidden_size, num_of_emotions)
        self.dropout_linear = nn.Dropout(p=0)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        conv_embedding = self.conv2Dblock(x)
        conv_embedding = torch.flatten(conv_embedding, start_dim=1)
        x_reduced = self.lstm_maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(0, 2, 1)
        lstm_embedding, (h, c) = self.lstm(x_reduced)
        attention_weights = torch.stack(
            [
                self.attention_linear(lstm_embedding[:, t, :])
                for t in range(lstm_embedding.size(1))
            ],
            dim=1,
        )
        attention_weights_norm = nn.functional.softmax(attention_weights, dim=1)
        attention = torch.bmm(attention_weights_norm.permute(0, 2, 1), lstm_embedding)
        attention = torch.squeeze(attention, 1)
        
        complete_embedding = torch.cat([conv_embedding, attention], dim=1)
        output_logits = self.out_linear(complete_embedding)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax
    
def loss_function(predictions, targets):
    return nn.CrossEntropyLoss()(predictions, targets)

